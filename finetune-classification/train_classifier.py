import os
import argparse
import math
import torch
import sys
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from tqdm import tqdm

sys.path.append("..")
from model import MAE_ViT, ViT_Classifier
from utils import setup_seed

def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    setup_seed(args.seed, args.determinism)

    # set batch_size and load_batch_size
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # data preprocessing
    transform_train = v2.Compose([
        v2.ToImage(),
        v2.Resize((args.input_size, args.input_size)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    transform_valid = v2.Compose([
        v2.ToImage(),
        v2.Resize((args.input_size, args.input_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_dataset = ImageFolder(root=args.data_root + '/train', transform=transform_train)
    valid_dataset = ImageFolder(root=args.data_root + '/valid', transform=transform_valid)

    # data augmentation
    num_class = len(train_dataset.classes)
    cutmix = v2.CutMix(num_classes=num_class)
    mixup = v2.MixUp(num_classes=num_class)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    # distribute data using DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=load_batch_size, sampler=train_sampler, num_workers=8, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=load_batch_size, sampler=valid_sampler, num_workers=8)

    # saving model and log
    if local_rank == 0:
        finetuned_dataset_name = os.path.basename(args.data_root)
        checkpoint_dir = os.path.join('checkpoints', finetuned_dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, args.pretrained_dataset_name+'_'+args.pretrained_model_name[4:]+'_classifier.pt')
        writer = SummaryWriter(os.path.join('logs', finetuned_dataset_name, args.pretrained_dataset_name+'_'+args.pretrained_model_name[4: ]+'_classifier'))

    # loading a pretrained model
    pretrained_model_path = os.path.join('../pretrain-ultrasound', 'checkpoints', 
                                         args.pretrained_dataset_name,
                                         args.pretrained_model_name+'.pt')
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')

    # initialize MAE_ViT
    if args.pretrained_model_name == 'mae_vit_base_patch16':
        mae_vit = MAE_ViT(image_size=args.input_size, patch_size=16,
                encoder_emb_dim=768, encoder_layer=12, encoder_head=12,
                decoder_emb_dim=768, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    elif args.pretrained_model_name == 'mae_vit_large_patch16':
        mae_vit = MAE_ViT(image_size=args.input_size, patch_size=16,
                encoder_emb_dim=1024, encoder_layer=24, encoder_head=16,
                decoder_emb_dim=1024, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    elif args.pretrained_model_name == 'mae_vit_huge_patch14':
        mae_vit = MAE_ViT(image_size=args.input_size, patch_size=14,
                encoder_emb_dim=1280, encoder_layer=32, encoder_head=16,
                decoder_emb_dim=1280, decoder_layer=8, decoder_head=16,
                mask_ratio=args.mask_ratio)
    else:
        print("pretrained model name not exists")
        exit(0)
        
    # extract state dictionary
    state_dict = checkpoint['model_state_dict']
    # remove 'module.' prefix if DDP
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()} \
        if any(k.startswith('module.') for k in state_dict.keys()) else state_dict

    # load state dictionary into MAE_ViT
    mae_vit.load_state_dict(state_dict, strict=False)
    model = ViT_Classifier(mae_vit.encoder, num_classes=num_class)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # define loss functions and evaluation metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    # define optimizer and learning rate scheduler
    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.base_learning_rate * args.batch_size / 256,
                              betas=(0.9, 0.999),
                              weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # train and validate the model
    best_val_loss = float('inf')
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        train_sampler.set_epoch(e)
        for image, label in tqdm(iter(train_dataloader)):
            step_count += 1
            image = image.cuda(local_rank)
            label = label.cuda(local_rank)
            logits = model(image)
            loss = loss_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        if local_rank == 0:
            print('In epoch {}, average training loss is {:.2f}.'.format(e, avg_train_loss))

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for image, label in tqdm(iter(valid_dataloader)):
                image = image.cuda(local_rank)
                label = label.cuda(local_rank)
                logits = model(image)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            if local_rank == 0:
                print('In epoch {}, average validation loss is {:.2f}, average validation acc is {:.2%}.'.format(e, avg_val_loss, avg_val_acc))

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            if local_rank == 0:
                print('saving best model with loss {:.2f} at {} epoch!'.format(best_val_loss, e))  
                torch.save(model, checkpoint_path)

        if local_rank == 0:
            writer.add_scalars('classification/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=e)
            writer.add_scalars('classification/acc', {'val' : avg_val_acc}, global_step=e)

    # cleaning up distributed environments
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_device_batch_size', type=int, default=64)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--pretrained_dataset_name', type=str, default='')
    parser.add_argument('--pretrained_model_name', type=str, default='')
    parser.add_argument('--determinism', action='store_true')

    args = parser.parse_args()

    main(args)