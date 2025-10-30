import torch
import torch.nn.functional as F


class MultiClassDiceCE(torch.nn.Module):
    def __init__(self, smooth=1e-5, ce_weight=1.0, dice_weight=1.0, num_class=2):
        super().__init__()
        self.smooth = smooth
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_class

    def forward(self, pred, target):
        # cross entropy loss
        ce_loss = F.cross_entropy(pred, target) * self.ce_weight
        
        # dice loss
        pred_probs = F.softmax(pred, dim=1)                                          # [batch_size, num_classes, input_size, input_size]
        target_oh = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()  # [batch_size, num_classes, input_size, input_size]
        
        intersection = torch.sum(pred_probs * target_oh, dim=(2, 3))
        union = torch.sum(pred_probs, dim=(2, 3)) + torch.sum(target_oh, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1 - dice.mean()) * self.dice_weight
        
        return ce_loss + dice_loss