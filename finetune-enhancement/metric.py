import torch
import numpy as np
import torch.nn.functional as F

from pytorch_msssim import ssim
from sklearn.metrics import normalized_mutual_info_score

def compute_LNCC(img1, img2, kernel_size=9):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")
    
    padding = kernel_size // 2
    img1 = F.pad(img1, [padding]*4, 'reflect')
    img2 = F.pad(img2, [padding]*4, 'reflect')

    # Compute local sums for img1, img2, img1^2, img2^2, and img1*img2
    sum_img1 = F.avg_pool2d(img1, kernel_size, stride=1)
    sum_img2 = F.avg_pool2d(img2, kernel_size, stride=1)
    sum_img1_sq = F.avg_pool2d(img1 * img1, kernel_size, stride=1)
    sum_img2_sq = F.avg_pool2d(img2 * img2, kernel_size, stride=1)
    sum_img1_img2 = F.avg_pool2d(img1 * img2, kernel_size, stride=1)
    
    # Compute mean and variance in local window for img1 and img2
    mean_img1 = sum_img1 / (kernel_size ** 2)
    mean_img2 = sum_img2 / (kernel_size ** 2)
    var_img1 = sum_img1_sq - mean_img1 ** 2
    var_img2 = sum_img2_sq - mean_img2 ** 2

    # Compute the local normalized cross-correlation
    ncc = (sum_img1_img2 - mean_img1 * mean_img2) / (torch.sqrt(var_img1 * var_img2) + 1e-5)
    
    return torch.mean(ncc)


def compute_metrics(pred, target):
    """Compute SSIM, PSNR, LNCC, and NMI between predicted and target images."""
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
    
    # LNCC
    lncc_val = compute_LNCC(pred, target, kernel_size=9).item()
    
    # NMI
    pred_np = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    pred_hist, _ = np.histogram(pred_np, bins=256, range=(0, 1))
    target_hist, _ = np.histogram(target_np, bins=256, range=(0, 1))
    nmi_val = normalized_mutual_info_score(target_hist, pred_hist, average_method='arithmetic')
    
    return ssim_val, psnr_val, lncc_val, nmi_val