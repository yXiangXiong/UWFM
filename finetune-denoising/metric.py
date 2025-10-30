import torch
import torch.nn.functional as F

from pytorch_msssim import ssim


def compute_metrics(pred, target):
    """Compute SSIM, PSNR between predicted and target images."""
    # Ensure images are in [0, 1] range
    pred = (pred + 1) / 2.0
    # print(f"Pred range: {pred.min().item():.3f} to {pred.max().item():.3f}")
    target = (target + 1) / 2.0
    # print(f"Target range: {target.min().item():.3f} to {target.max().item():.3f}")
    
    # SSIM
    ssim_val = ssim(pred.unsqueeze(0), target.unsqueeze(0), data_range=1.0, size_average=True, nonnegative_ssim=True).item()
    
    # PSNR
    mse = F.mse_loss(pred, target)
    psnr_val = (10 * torch.log10(1 / mse)).item() if mse > 0 else 100.0
    
    return ssim_val, psnr_val