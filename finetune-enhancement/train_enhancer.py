import os
import sys
import torch
import lpips
import argparse
import warnings
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch import nn

sys.path.append("..")
from cyclegan import Discriminator, Generator, GeneratorWithMAE
from dataset import EnhancementDataset
from model import MAE_ViT
from metric import compute_LNCC
from utils import setup_seed
from pytorch_msssim import ssim
warnings.filterwarnings('ignore')


def train_fn(discHigh, discLow, genH2L, genL2H, loader, opt_disc, opt_gen, l1, mse, perLoss, d_scaler, g_scaler, local_rank, args):
    H_reals = 0
    H_fakes = 0
    DiscLoss = 0
    GenLoss = 0
    loop = tqdm(loader, leave=True) if local_rank == 0 else loader
    for idx, (low, high, _) in enumerate(loop):
        low = low.cuda(local_rank)
        high = high.cuda(local_rank)

        # Validate input data
        if local_rank == 0:
            if torch.isnan(low).any() or torch.isnan(high).any():
                print(f"Warning: NaN detected in input data at batch {idx}")
                continue

        with torch.cuda.amp.autocast():
            fake_high = genL2H(low)
            D_H_real = discHigh(high)
            D_H_fake = discHigh(fake_high.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            factor = 0.9  # Reduced factor for stability (label smoothing)
            D_H_loss = mse(D_H_real, torch.ones_like(D_H_real) * factor) + mse(D_H_fake, torch.zeros_like(D_H_fake))

            fake_low = genH2L(high)
            D_L_real = discLow(low)
            D_L_fake = discLow(fake_low.detach())
            factor = 0.9  # Consistent factor
            D_L_loss = mse(D_L_real, torch.ones_like(D_L_real) * factor) + mse(D_L_fake, torch.zeros_like(D_L_fake))
            D_loss = D_H_loss + D_L_loss

            # Check for NaN in discriminator loss
            if torch.isnan(D_loss):
                print(f"NaN detected in D_loss at batch {idx}")
                print(f"D_H_real: {D_H_real.mean().item()}, D_H_fake: {D_H_fake.mean().item()}")
                print(f"D_L_real: {D_L_real.mean().item()}, D_L_fake: {D_L_fake.mean().item()}")
                continue

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        # Gradient clipping for discriminator
        nn.utils.clip_grad_norm_(list(discHigh.parameters()) + list(discLow.parameters()), max_norm=1.0)
        d_scaler.step(opt_disc)
        d_scaler.update()
        DiscLoss += D_loss.item() if not torch.isnan(D_loss) else 0

        with torch.cuda.amp.autocast():
            fake_high = genL2H(low)
            fake_low = genH2L(high)
            D_H_fake = discHigh(fake_high)
            D_L_fake = discLow(fake_low)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))

            cycle_low = genH2L(fake_high)
            cycle_high = genL2H(fake_low)
            cycleLow = l1(low, cycle_low)
            cycleHigh = l1(high, cycle_high)

            identity_low = genH2L(low)
            identity_high = genL2H(high)
            identityLow = l1(low, identity_low)
            identityHigh = l1(high, identity_high)

            G_loss = ((loss_G_L + loss_G_H) * args.lambda_adv + 
                      (cycleLow + cycleHigh) * args.lambda_cycle + 
                      (identityLow + identityHigh) * args.lambda_identity)

            if torch.isnan(G_loss):
                print(f"NaN detected in G_loss at batch {idx}")
                continue

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        # Gradient clipping for generator
        nn.utils.clip_grad_norm_(list(genH2L.parameters()) + list(genL2H.parameters()), max_norm=1.0)
        g_scaler.step(opt_gen)
        g_scaler.update()
        GenLoss += G_loss.item() if not torch.isnan(G_loss) else 0

        if local_rank == 0:
            loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1), D_loss=DiscLoss / (idx + 1), G_loss=GenLoss / (idx + 1))


def val_fn(Gen, val_loader, mse, local_rank):
    Gen.eval()
    psnrScore = 0.0
    lnccScore = 0.0
    ssimScore = 0.0
    highL1 = 0.0

    with torch.no_grad():
        for idx, (low, high, _) in enumerate(val_loader):
            low = low.cuda(local_rank)
            high = high.cuda(local_rank)
            fake_high = Gen(low)

            ssimScore += ssim((fake_high+1)/2.0, (high+1)/2.0, data_range=1.0, size_average=True, nonnegative_ssim=True).item()
            lnccScore += compute_LNCC((fake_high+1)/2, (high+1)/2, kernel_size=9).item()
            mseVal = mse((fake_high+1)/2.0, (high+1)/2.0)
            psnrScore += (10 * torch.log10(1/(mseVal+1e-8))).item() if (mseVal > 0) else 100
            highL1 += torch.mean(torch.abs(high-fake_high))

    Gen.train()
    if local_rank == 0:
        print(f"SSIM: {ssimScore/len(val_loader):.6f}, LNCC: {lnccScore/len(val_loader):.6f}, PSNR: {psnrScore/len(val_loader):.6f}, HighL1: {highL1/len(val_loader):.6f}")
    return (ssimScore/len(val_loader), lnccScore/len(val_loader), psnrScore/len(val_loader), highL1/len(val_loader))


def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    setup_seed(args.seed, args.determinism)
    
    # data preprocessing
    train_dataset = EnhancementDataset(os.path.join(args.data_root, 'train'), args.input_size, is_dual_transform=True)
    valid_dataset = EnhancementDataset(os.path.join(args.data_root, 'valid'), args.input_size)

    # distribute data using DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=8)

    # save the best generator
    finetuned_dataset_name = args.data_root.split('/')[-1]
    checkpoint_dir = os.path.join('checkpoints', finetuned_dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.baseline_cyclegan:
        checkpoint_Best_G_L2H_path = os.path.join(checkpoint_dir, "cyclegan_G_L2H.pt")
    else:
        checkpoint_Best_G_L2H_path = os.path.join(checkpoint_dir, f"{args.pretrained_dataset_name}_{args.pretrained_model_name[4:]}_G_L2H.pt")

    # load a pretrained model
    pretrained_model_path = os.path.join('../pretrain-ultrasound', 'checkpoints',
                                         args.pretrained_dataset_name,
                                         f"{args.pretrained_model_name}.pt")
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')

    if not args.baseline_cyclegan:
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

    # initialize cyclegan modules
    discHigh = Discriminator().cuda(local_rank)
    discHigh = DDP(discHigh, device_ids=[local_rank])
    discLow = Discriminator().cuda(local_rank)
    discLow = DDP(discLow, device_ids=[local_rank])

    if args.baseline_cyclegan:
        genH2L = Generator(num_channels=3).cuda(local_rank)
        genH2L = DDP(genH2L, device_ids=[local_rank])
        genL2H = Generator(num_channels=3).cuda(local_rank)
        genL2H = DDP(genL2H, device_ids=[local_rank])
    else: # load state dictionary into MAE_ViT
        mae_vit.load_state_dict(state_dict, strict=False)
        if args.pretrained_model_name == 'mae_vit_base_patch16':
            genH2L = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=768,
                                      num_channels=3).cuda(local_rank)
            genL2H = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=768,
                                      num_channels=3).cuda(local_rank)
        if args.pretrained_model_name == 'mae_vit_large_patch16':
            genH2L = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1024,
                                      num_channels=3).cuda(local_rank)
            genL2H = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1024,
                                      num_channels=3).cuda(local_rank)
        if args.pretrained_model_name == 'mae_vit_huge_patch14':
            genH2L = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1280,
                                      num_channels=3).cuda(local_rank)
            genL2H = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1280,
                                      num_channels=3).cuda(local_rank)
        genH2L = DDP(genH2L, device_ids=[local_rank])
        genL2H = DDP(genL2H, device_ids=[local_rank])
    
    # optimizers with reduced learning rate for discriminator
    opt_disc = optim.Adam(list(discHigh.parameters()) + list(discLow.parameters()), lr=args.lr_disc * 0.1, betas=(0.9, 0.9))
    opt_gen = optim.Adam(list(genH2L.parameters()) + list(genL2H.parameters()), lr=args.lr_gen, betas=(0.9, 0.9))

    # learning rate schedulers
    lr_scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=args.lr_step, gamma=args.lr_gamma)
    lr_scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=args.lr_step, gamma=args.lr_gamma)

    # loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    perLoss = lpips.LPIPS(net='vgg').cuda(local_rank)
    
    # gradient scalers for mixed precision
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # training loop
    ssimVals, lnccVals, psnrVals, highL1Vals = [], [], [], []
    bestHighL1 = 100.0

    for epoch in range(args.num_epochs):
        if local_rank == 0:  # Only print from rank 0
            print(f"Epoch [{epoch}/{args.num_epochs}]")
        train_sampler.set_epoch(epoch)
        train_fn(discHigh, discLow,
                 genH2L, genL2H,
                 train_loader,
                 opt_disc, opt_gen,
                 L1, mse, perLoss,
                 d_scaler, g_scaler,
                 local_rank, args)

        (ssimScore, lnccScore, psnrScore, highL1) = val_fn(genL2H, val_loader, mse, local_rank)
        
        lr_scheduler_disc.step()
        lr_scheduler_gen.step()

        ssimVals.append(ssimScore)
        lnccVals.append(lnccScore)
        psnrVals.append(psnrScore)
        highL1Vals.append(highL1.cpu().numpy())
        
        if highL1 < bestHighL1:
            bestHighL1, bestEPOCH = highL1, epoch
            if local_rank == 0:
                print(f"Save best generator at epoch: {epoch} !")
                torch.save(genL2H, checkpoint_Best_G_L2H_path)
            
    print("Done training!")
    if local_rank == 0:
        print(f"Best highL1: {bestHighL1:.6f}, Best Epoch: {bestEPOCH}")

    # cleaning up distributed environments
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN-based Image Enhancement")
    parser.add_argument('--data_root', type=str, default='', help='Root directory for dataset')
    parser.add_argument('--input_size', type=int, default=224, help='Size of input images')

    parser.add_argument('--pretrained_dataset_name', type=str, default='', help='Name of pretrained dataset')
    parser.add_argument('--pretrained_model_name', type=str, default='', help='Name of pretrained model')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--determinism', action='store_true', help='Ensure deterministic training')
    parser.add_argument('--baseline_cyclegan', action='store_true', help='use baseline cyclegan or not')

    parser.add_argument('--lr_gen', type=float, default=2e-4, help='Learning rate for generators')
    parser.add_argument('--lr_disc', type=float, default=2e-4, help='Learning rate for discriminators')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--lr_step', type=int, default=50, help='Steps for learning rate decay')

    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_identity', type=float, default=1.0, help='Identity loss weight')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle consistency loss weight')
    parser.add_argument('--lambda_per', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    args = parser.parse_args()
    if not args.data_root or not args.pretrained_dataset_name or not args.pretrained_model_name:
        raise ValueError("Please specify --data_root, --pretrained_dataset_name, and --pretrained_model_name")
    
    main(args)