import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def calculate_dice(pred, target, num_class, smooth=1e-6):
    """
    Calculate Dice score for multi-class segmentation.
    
    Args:
        pred: Predicted mask tensor [batch_size, H, W], values in [0, num_class-1]
        target: Ground truth mask tensor [batch_size, H, W], values in [0, num_class-1]
        num_class: Number of classes
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Average Dice score across all classes
    """
    dice_scores = []
    pred = pred.view(-1)  # 展平为 1D
    target = target.view(-1)
    
    for cls in range(num_class):
        pred_cls = (pred == cls).float()      # 当前类别的二值化预测
        target_cls = (target == cls).float()  # 当前类别的二值化真实值
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)
    
    return torch.mean(torch.tensor(dice_scores))


def calculate_iou(pred, target, num_class, smooth=1e-6):
    """Calculate IoU (Jaccard Index) for multi-class segmentation."""
    iou_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_class):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)
    
    return torch.mean(torch.tensor(iou_scores))

def calculate_accuracy(pred, target):
    """Calculate pixel-wise accuracy."""
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

def calculate_sensitivity(pred, target, num_class, smooth=1e-6):
    """Calculate sensitivity (recall) for multi-class segmentation."""
    sensitivity_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_class):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        true_positives = (pred_cls * target_cls).sum()
        actual_positives = target_cls.sum()
        
        sensitivity = (true_positives + smooth) / (actual_positives + smooth)
        sensitivity_scores.append(sensitivity)
    
    return torch.mean(torch.tensor(sensitivity_scores))

def calculate_hd95(pred, target, num_class):
    """Calculate 95th percentile Hausdorff Distance for multi-class segmentation."""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    hd_scores = []
    
    for cls in range(num_class):
        pred_cls = (pred == cls).astype(np.uint8)
        target_cls = (target == cls).astype(np.uint8)
        
        # 计算 Hausdorff 距离
        if pred_cls.sum() == 0 or target_cls.sum() == 0:
            hd_scores.append(np.inf if pred_cls.sum() != target_cls.sum() else 0)
        else:
            hd_forward = directed_hausdorff(pred_cls, target_cls)[0]
            hd_backward = directed_hausdorff(target_cls, pred_cls)[0]
            hd = max(hd_forward, hd_backward)
            # 计算 95th percentile（近似）
            hd_scores.append(hd)
    
    # 取有限值的平均值
    hd_scores = [x for x in hd_scores if np.isfinite(x)]
    if not hd_scores:
        return np.inf
    return np.percentile(hd_scores, 95)