import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True):
        """
        SSIM Loss模块
        参数：
            window_size: 滑动窗口大小（必须为奇数）
            channel: 输入图像的通道数
            size_average: 是否对loss取平均
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())

    def _create_gaussian_kernel(self):
        """
        创建高斯卷积核
        """
        sigma = 1.5  # 与原始SSIM论文设置一致
        coords = torch.arange(self.window_size).float()
        coords -= (self.window_size - 1) / 2.0
        g = coords**2 / (2 * sigma**2)
        g = torch.exp(-g)
        g = g / g.sum()  # 归一化
        g = g.view(1, 1, self.window_size, 1)  # 扩展为2D高斯核
        return g * g.permute(0,1,3,2)  # 转置后相乘得到二维高斯核

    def ssim(self, img1, img2):
        """
        计算SSIM值（返回1 - SSIM作为loss）
        """
        # 确保输入在[0,1]范围内
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        
        # 扩展高斯核维度
        kernel = self.gaussian_kernel.repeat(self.channel, 1, 1, 1)
        
        # 计算均值
        mu1 = F.conv2d(img1, kernel, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, kernel, padding=self.window_size//2, groups=self.channel)
        
        # 计算方差和协方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, kernel, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, kernel, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, kernel, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        # SSIM常数
        C1 = (0.01)**2
        C2 = (0.03)**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

    def forward(self, denoised, clean):
        """
        前向传播
        参数：
            denoised: 去噪后的图像 (B,C,H,W) 范围[0,1]
            clean: 干净图像 (B,C,H,W) 范围[0,1]
        """
        return self.ssim(denoised, clean)