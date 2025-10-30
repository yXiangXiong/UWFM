import sys
import torch
import torch.nn.functional as F

from torch import nn
from model import MAE_Encoder
from einops import rearrange

sys.path.append("..")


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.norm3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode='reflect')),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.norm2(x)
        x = self.norm3(x)
        x = self.conv2(x)

        return x


class Generator(nn.Module):
    def __init__(self, num_channels):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
        nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True))
        
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(9)])

        self.conv4 = nn.Sequential(
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
        nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True))

        self.conv6 = nn.Conv2d(128, num_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = self.conv3(x2)    

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        x = torch.cat([x, x2], dim=1)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.tanh(self.conv6(x))
        
        return x
    

class GeneratorWithMAE(nn.Module):
    def __init__(self,
                 encoder: MAE_Encoder,
                 image_size=224,
                 patch_size=16,
                 encoder_emb_dim=1024,
                 num_channels=3):
        super().__init__()

        # inherit some components of MAE Encoder
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm

        self.image_size = image_size
        self.patch_size = patch_size

        # cls feature projection layer
        self.cls_proj = nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.Conv2d(encoder_emb_dim, 64, 1),
                nn.Upsample(scale_factor=224 // (image_size // patch_size), mode='bilinear')),
            'conv2': nn.Sequential(
                nn.Conv2d(encoder_emb_dim, 128, 1),
                nn.Upsample(scale_factor=112 // (image_size // patch_size), mode='bilinear')),
            'conv3': nn.Sequential(
                nn.Conv2d(encoder_emb_dim, 256, 1),
                nn.Upsample(scale_factor=56 // (image_size // patch_size), mode='bilinear'))
        })
        
        # feature fusion layer (1x1 convolution compression channel)
        self.fusion_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(64*2, 64, 1),
            'conv2': nn.Conv2d(128*2, 128, 1),
            'conv3': nn.Conv2d(256*2, 256, 1)
        })

        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True))
        
        # residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(9)])

        # decoder
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Conv2d(128, num_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x):
        patches = self.patchify(x)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.transformer(patches)
        features = self.layer_norm(features)
        cls_feature = features[:, 0, :]
        B, C = cls_feature.shape
        cls_feature = cls_feature.view(B, C, 1, 1)
        cls_feature = F.interpolate(cls_feature,
                                    size=(self.image_size//self.patch_size, self.image_size//self.patch_size),
                                    mode='bilinear')

        x1 = self.conv1(x)
        proj_cls1 = self.cls_proj['conv1'](cls_feature)
        x1 = torch.cat([x1, proj_cls1], dim=1)       
        x1 = self.fusion_conv['conv1'](x1)               

        x2 = self.conv2(x1)
        proj_cls2 = self.cls_proj['conv2'](cls_feature)
        x2 = torch.cat([x2, proj_cls2], dim=1)
        x2 = self.fusion_conv['conv2'](x2)          

        x = self.conv3(x2)
        proj_cls3 = self.cls_proj['conv3'](cls_feature)
        x = torch.cat([x, proj_cls3], dim=1)
        x = self.fusion_conv['conv3'](x)            

        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv4(x)
        x = torch.cat([x, x2], dim=1)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.tanh(self.conv6(x))
        
        return x