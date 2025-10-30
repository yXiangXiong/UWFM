import os
import sys
import torch
import lpips
import argparse
import warnings
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

sys.path.append("..")
from cyclegan import Discriminator, Generator, GeneratorWithMAE
from dataset import DenoisingDataset
from model import MAE_ViT
from loss import SSIMLoss
from utils import setup_seed
from pytorch_msssim import ssim
warnings.filterwarnings('ignore')


def train_fn(discClean, discNoised, genC2N, genN2C, loader, opt_disc, opt_gen, l1, mse, perLoss, ssim_loss, d_scaler, g_scaler, local_rank, args):
    Clean_reals = 0
    Clean_fakes = 0
    DiscLoss = 0
    GenLoss = 0
    loop = tqdm(loader, leave=True) if local_rank == 0 else loader
    for idx, (noised_image, clean_image, _) in enumerate(loop):
        noised_image = noised_image.cuda(local_rank)
        clean_image = clean_image.cuda(local_rank)

        # Validate input data
        if torch.isnan(noised_image).any() or torch.isnan(clean_image).any():
            print(f"Warning: NaN detected in input data at batch {idx}")
            continue

        with torch.cuda.amp.autocast():
            fake_clean = genN2C(noised_image)
            D_Clean_real = discClean(clean_image)
            D_Clean_fake = discClean(fake_clean.detach())
            Clean_reals += D_Clean_real.mean().item()
            Clean_fakes += D_Clean_fake.mean().item()
            factor = 0.9  # Reduced factor for stability (label smoothing)
            D_Clean_loss = mse(D_Clean_real, torch.ones_like(D_Clean_real) * factor) + mse(D_Clean_fake, torch.zeros_like(D_Clean_fake))

            fake_noised = genC2N(clean_image)
            D_Noised_real = discNoised(noised_image)
            D_Noised_fake = discNoised(fake_noised.detach())
            factor = 0.9  # Consistent factor
            D_Noised_loss = mse(D_Noised_real, torch.ones_like(D_Noised_real) * factor) + mse(D_Noised_fake, torch.zeros_like(D_Noised_fake))

            D_loss = D_Clean_loss + D_Noised_loss

            # Check for NaN in discriminator loss
            if torch.isnan(D_loss):
                print(f"NaN detected in D_loss at batch {idx}")
                print(f"D_Clean_real: {D_Clean_real.mean().item()}, D_Clean_fake: {D_Clean_fake.mean().item()}")
                print(f"D_Noised_real: {D_Noised_real.mean().item()}, D_Noised_fake: {D_Noised_fake.mean().item()}")
                continue

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        # Gradient clipping for discriminator
        nn.utils.clip_grad_norm_(list(discClean.parameters()) + list(discNoised.parameters()), max_norm=1.0)
        d_scaler.step(opt_disc)
        d_scaler.update()
        DiscLoss += D_loss.item() if not torch.isnan(D_loss) else 0

        with torch.cuda.amp.autocast():
            fake_clean = genN2C(noised_image)
            fake_noised = genC2N(clean_image)
            D_Clean_fake = discClean(fake_clean)
            D_Noised_fake = discNoised(fake_noised)
            loss_G_Clean = mse(D_Clean_fake, torch.ones_like(D_Clean_fake))
            loss_G_Noised = mse(D_Noised_fake, torch.ones_like(D_Noised_fake))

            cycle_noised = genC2N(fake_clean)
            cycle_clean = genN2C(fake_noised)
            cycleNoised = l1(noised_image, cycle_noised)
            cycleClean = l1(clean_image, cycle_clean)

            identity_noised = genN2C(noised_image)
            identity_clean = genC2N(clean_image)
            identityNoised = l1(noised_image, identity_noised)
            identityClean = l1(clean_image, identity_clean)

            perNoised = torch.mean(perLoss.forward(noised_image, fake_noised))
            perClean = torch.mean(perLoss.forward(clean_image, fake_clean))

            ssim_noised = torch.mean(ssim_loss((noised_image+1)/2.0, (fake_noised+1)/2.0))
            ssim_clean = torch.mean(ssim_loss((clean_image+1)/2.0, (fake_clean+1)/2.0))

            G_loss = ((loss_G_Noised + loss_G_Clean) * args.lambda_adv + 
                      (cycleNoised + cycleClean) * args.lambda_cycle + 
                      (identityNoised + identityClean) * args.lambda_identity + 
                      (perNoised + perClean) * args.lambda_per + 
                      (ssim_noised + ssim_clean) * args.lambda_ssim
                      )

            if torch.isnan(G_loss):
                print(f"NaN detected in G_loss at batch {idx}")
                continue

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        # Gradient clipping for generator
        nn.utils.clip_grad_norm_(list(genC2N.parameters()) + list(genN2C.parameters()), max_norm=1.0)
        g_scaler.step(opt_gen)
        g_scaler.update()
        GenLoss += G_loss.item() if not torch.isnan(G_loss) else 0

        if local_rank == 0:
            loop.set_postfix(Clean_real=Clean_reals / (idx + 1), Clean_fake=Clean_fakes / (idx + 1), D_loss=DiscLoss / (idx + 1), G_loss=GenLoss / (idx + 1))


def val_fn(Gen, val_loader, mse, local_rank):
    Gen.eval()
    psnrScore = 0.0
    ssimScore = 0.0
    cleanL1 = 0.0

    with torch.no_grad():
        for idx, (noised_image, clean_image, _) in enumerate(val_loader):
            noised_image = noised_image.cuda(local_rank)
            clean_image = clean_image.cuda(local_rank)
            fake_clean = Gen(noised_image)

            ssimScore += ssim((fake_clean+1)/2.0, (clean_image+1)/2.0, data_range=1.0, size_average=True, nonnegative_ssim=True).item()
            mseVal = mse((fake_clean+1)/2.0, (clean_image+1)/2.0)
            psnrScore += (10 * torch.log10(1/mseVal)).item() if (mseVal > 0) else 100
            cleanL1 += torch.mean(torch.abs(clean_image-fake_clean))

    Gen.train()
    if local_rank == 0:
        print(f"SSIM: {ssimScore/len(val_loader):.6f}, PSNR: {psnrScore/len(val_loader):.6f}, CleanL1: {cleanL1/len(val_loader):.6f}")
    return (ssimScore/len(val_loader), psnrScore/len(val_loader), cleanL1/len(val_loader))


def main(args):
    # get local_rank from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    print(f"Process {local_rank} is using GPU {torch.cuda.current_device()}")

    # initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    setup_seed(args.seed, args.determinism)

    # data preprocessing
    train_dataset = DenoisingDataset(os.path.join(args.data_root, 'train'), args.input_size, is_dual_transform=True)
    valid_dataset = DenoisingDataset(os.path.join(args.data_root, 'valid'), args.input_size)
    
    # distribute data using DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=8)

    # Save the best generator
    finetuned_dataset_name = args.data_root.split('/')[-1]
    checkpoint_dir = os.path.join('checkpoints', finetuned_dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.baseline_cyclegan:
        checkpoint_Best_G_N2C_path = os.path.join(checkpoint_dir, "cyclegan_G_N2C.pt")
    else:
        checkpoint_Best_G_N2C_path = os.path.join(checkpoint_dir, f"{args.pretrained_dataset_name}_{args.pretrained_model_name[4:]}_G_N2C.pt")

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
    discClean = Discriminator().cuda(local_rank)
    discClean = DDP(discClean, device_ids=[local_rank])
    discNoised = Discriminator().cuda(local_rank)
    discNoised = DDP(discNoised, device_ids=[local_rank])

    if args.baseline_cyclegan:
        genC2N = Generator(num_channels=3).cuda(local_rank)
        genC2N = DDP(genC2N, device_ids=[local_rank])
        genN2C = Generator(num_channels=3).cuda(local_rank)
        genN2C = DDP(genN2C, device_ids=[local_rank])
    else: # masked auto-encoder basedcyclegan
        # load state dictionary into MAE_ViT
        mae_vit.load_state_dict(state_dict, strict=False)
        if args.pretrained_model_name == 'mae_vit_base_patch16':
            genC2N = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=768,
                                      num_channels=3).cuda(local_rank)
            genN2C = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=768,
                                      num_channels=3).cuda(local_rank)
        if args.pretrained_model_name == 'mae_vit_large_patch16':
            genC2N = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1024,
                                      num_channels=3).cuda(local_rank)
            genN2C = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1024,
                                      num_channels=3).cuda(local_rank)
        if args.pretrained_model_name == 'mae_vit_huge_patch14':
            genC2N = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1280,
                                      num_channels=3).cuda(local_rank)
            genN2C = GeneratorWithMAE(mae_vit.encoder,
                                      image_size=args.input_size,
                                      patch_size=16,
                                      encoder_emb_dim=1280,
                                      num_channels=3).cuda(local_rank)

        genC2N = DDP(genC2N, device_ids=[local_rank])
        genN2C = DDP(genN2C, device_ids=[local_rank])
    
    # optimizers with reduced learning rate for discriminator
    opt_disc = optim.Adam(list(discClean.parameters()) + list(discNoised.parameters()), lr=args.lr_disc * 0.1, betas=(0.9, 0.9))
    opt_gen = optim.Adam(list(genC2N.parameters()) + list(genN2C.parameters()), lr=args.lr_gen, betas=(0.9, 0.9))

    # learning rate schedulers
    lr_scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=args.lr_step, gamma=args.lr_gamma)
    lr_scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=args.lr_step, gamma=args.lr_gamma)

    # loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    ssim_loss = SSIMLoss().cuda(local_rank)
    perLoss = lpips.LPIPS(net='vgg').cuda(local_rank)
    
    # gradient scalers for mixed precision
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # training loop
    ssimVals, psnrVals, cleanL1Vals = [], [], []
    bestSSIM, bestPSNR = 0.0, 0.0

    for epoch in range(args.num_epochs):
        if local_rank == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}]")
        train_sampler.set_epoch(epoch)
        train_fn(discClean, discNoised,
                 genC2N, genN2C,
                 train_loader,
                 opt_disc, opt_gen,
                 L1, mse, ssim_loss, perLoss,
                 d_scaler, g_scaler,
                 local_rank, args)
        (ssimScore, psnrScore, cleanL1) = val_fn(genN2C, val_loader, mse, local_rank)
        
        lr_scheduler_disc.step()
        lr_scheduler_gen.step()

        ssimVals.append(ssimScore)
        psnrVals.append(psnrScore)
        cleanL1Vals.append(cleanL1.cpu().numpy())
        
        current_scores = {'SSIM': ssimScore, 'PSNR': psnrScore}
        best_scores = {'SSIM': bestSSIM, 'PSNR': bestPSNR}
        impr = {k: current_scores[k] > best_scores[k] for k in best_scores}

        if sum(impr.values()) >= 2:
            bestSSIM, bestPSNR, bestEPOCH = ssimScore, psnrScore, epoch
            if local_rank == 0:
                print(f"Save best generator at epoch: {epoch} !")
                torch.save(genN2C, checkpoint_Best_G_N2C_path)  # Save state_dict for portability
                
    
    print("Done training!")
    if local_rank == 0:
        print(f"Best SSIM: {bestSSIM:.6f}, Best PSNR: {bestPSNR:.6f}, Best Epoch: {bestEPOCH}")

    # cleaning up distributed environments
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN-based Image Enhancement")
    parser.add_argument('--data_root', type=str, default='', help='Root directory for dataset')
    parser.add_argument('--input_size', type=int, default=224, help='Size of input images')

    parser.add_argument('--pretrained_dataset_name', type=str, default='', help='Name of pretrained dataset')
    parser.add_argument('--pretrained_model_name', type=str, default='', help='Name of pretrained model')

    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--determinism', action='store_true', help='Ensure deterministic training')
    parser.add_argument('--baseline_cyclegan', action='store_true', help='use baseline cyclegan or not')

    parser.add_argument('--lr_gen', type=float, default=3e-4, help='Learning rate for generators')
    parser.add_argument('--lr_disc', type=float, default=3e-3, help='Learning rate for discriminators')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--lr_step', type=int, default=50, help='Steps for learning rate decay')

    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_identity', type=float, default=1.0, help='Identity loss weight')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle consistency loss weight')
    parser.add_argument('--lambda_per', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=1.0, help='SSIM loss weight')
    parser.add_argument('--mask_ratio', type=float, default=0.75)

    args = parser.parse_args()
    if not args.data_root or not args.pretrained_dataset_name or not args.pretrained_model_name:
        raise ValueError("Please specify --data_root, --pretrained_dataset_name, and --pretrained_model_name")
    
    main(args)