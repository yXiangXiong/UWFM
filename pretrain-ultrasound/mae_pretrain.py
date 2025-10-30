import os
import argparse
import math
import torch
import sys
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append("..")
from model import *
from uncertainty import *
from loss import residual_distribution
from utils import setup_seed


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, scaler):
    """load checkpoint and return the epoch to resume from."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch")
        return 0


def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    setup_seed(args.seed, args.determinism)

    # set batch_size and load_batch_size
    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    assert args.batch_size % load_batch_size == 0
    steps_per_update = args.batch_size // load_batch_size

    # data preprocessing
    transform_train = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.2, 1.0), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_valid = v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(args.input_size, args.input_size), interpolation=v2.InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = ImageFolder(root=args.dataset_root + '/train', transform=transform_train)
    valid_dataset = ImageFolder(root=args.dataset_root + '/valid', transform=transform_valid)

    # distributing data using DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=load_batch_size, sampler=train_sampler, num_workers=8)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, sampler=valid_sampler, num_workers=8)

    # saving model and log
    pretrained_dataset_name = os.path.basename(args.dataset_root)
    checkpoint_dir = os.path.join('checkpoints', pretrained_dataset_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, args.model_name+'.pt')
    if local_rank == 1:
        writer = SummaryWriter(os.path.join('logs', pretrained_dataset_name, args.model_name))

    # model initialization
    if args.model_name == 'mae_vit_base_patch16':
        model = MAE_ViT(image_size=args.input_size, patch_size=16,
                encoder_emb_dim=768, encoder_layer=12, encoder_head=12,
                decoder_emb_dim=768, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    elif args.model_name == 'mae_vit_large_patch16':
        model = MAE_ViT(image_size=args.input_size, patch_size=16,
                encoder_emb_dim=1024, encoder_layer=24, encoder_head=16,
                decoder_emb_dim=1024, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    elif args.model_name == 'mae_vit_huge_patch14':
        model = MAE_ViT(image_size=args.input_size, patch_size=14,
                encoder_emb_dim=1280, encoder_layer=32, encoder_head=16,
                decoder_emb_dim=1280, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    else:
        print("model name not exists")
        exit(0)

    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # define optimizer and learning rate scheduler
    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.base_learning_rate * args.batch_size / 256, 
                              betas=(0.9, 0.95), weight_decay=args.weight_decay
                              )
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1)
                                )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # initialize GradScaler for AMP
    scaler = GradScaler()

    # load checkpoint if exists
    start_epoch = load_checkpoint(checkpoint_path, model.module, optim, lr_scheduler, scaler)

    # train and validate the model
    step_count = 0
    optim.zero_grad()
    for e in range(start_epoch, args.total_epoch):
        train_sampler.set_epoch(e)
        model.train()
        losses = []
        for image, label in tqdm(iter(train_dataloader)):
            step_count += 1
            image = image.cuda(local_rank)
            
            # AMP context
            with autocast():
                pred_image, mask, alpha, beta = model(image)
                sigma = calculate_variance(alpha, beta)
                uncertainty_map = scaled_sigma(sigma)

                loss = torch.mean((pred_image - image) ** 2 * mask * uncertainty_map) + \
                        args.lam_rd * residual_distribution(pred_image, alpha, beta, image, mask)
            
            # scale the loss and backpropagate
            scaler.scale(loss).backward()
            
            if step_count % steps_per_update == 0:
                # unscales gradients and steps optimizer
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            losses.append(loss.item())
        
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        
        if local_rank == 1:
            writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average training loss is {avg_loss}.')

        model.eval()
        with torch.no_grad():
            for val_image, label in tqdm(iter(valid_dataloader)):
                val_image = val_image.cuda(local_rank)
                with autocast():
                    pred_val_image, mask, alpha, beta = model(val_image)
                    pred_val_image = pred_val_image * mask + val_image * (1 - mask)

                    sigma = calculate_variance(alpha, beta)
                    color_sigma = color_mapping(sigma)
                    color_alpha = color_mapping(alpha)
                    color_beta =  color_mapping(beta)
                    
                image = torch.cat([val_image, val_image * (1 - mask), pred_val_image,
                                   color_alpha, color_beta, color_sigma], dim=0)
                image = rearrange(image, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=6)
                if local_rank == 1:
                    if (e + 1) % 30 == 0:  # visualize every 20 epochs instead of every epoch
                        writer.add_image('mae_image', (image + 1) / 2, global_step=e)

        if (e + 1) % args.save_freq == 0:
            if local_rank == 0:
                checkpoint_epoch_name = os.path.basename(checkpoint_path).split('.')[0] + '_epoch{}.pt'.format(e+1)
                checkpoint_epoch_path = os.path.join(checkpoint_dir, checkpoint_epoch_name)
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_epoch_path)

    if local_rank == 1:
        torch.save({
            'epoch': args.total_epoch - 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, checkpoint_path)
    
    # cleaning up distributed environments
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='')
    parser.add_argument('--model_name', type=str, default='', help='Name of model to train')
    
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4096, help='control AdamW learning rate')
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--valid_batch_size', type=int, default=32)

    parser.add_argument('--total_epoch', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=25)

    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--lam_rd', type=float, default=0.1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--determinism', action='store_true')
    
    args = parser.parse_args()

    main(args)