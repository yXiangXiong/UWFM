import torch
import numpy as np
import torch.nn.functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape # T is number of patches
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        visible_patches = patches[:remain_T]

        return visible_patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=1024,
                 num_layer=24,
                 num_head=16,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=512,
                 num_layer=8,
                 num_head=16,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
                                torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.mean_head = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2 * emb_dim, 3 * patch_size ** 2)
                                             )
        self.alpha_head = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(2 * emb_dim, 1 * patch_size ** 2),
                                              torch.nn.ReLU())
        self.beta_head = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2 * emb_dim, 1 * patch_size ** 2),
                                             torch.nn.ReLU())

        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', 
                                   p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)


    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
                            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                             backward_indexes + 1], 
                             dim=0)
        features = torch.cat(
                        [features, 
                        self.mask_token.expand(backward_indexes.shape[0]-features.shape[0],
                        features.shape[1], -1)],
                        dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]                

        mean_patches = self.mean_head(features)
        alpha_patches = self.alpha_head(features)
        beta_patches = self.beta_head(features)

        mask = torch.zeros_like(mean_patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:]-1)
        img = self.patch2img(mean_patches)
        mask = self.patch2img(mask)

        alpha = self.patch2img(alpha_patches)
        beta = self.patch2img(beta_patches)
    
        return img, mask, alpha, beta
    

class MAE_ViT(torch.nn.Module):
    def __init__(self, image_size=224, patch_size=16,
                 encoder_emb_dim=1024, encoder_layer=24, encoder_head=16,
                 decoder_emb_dim=512, decoder_layer=8,  decoder_head=16,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size,
                                   encoder_emb_dim,
                                   encoder_layer, encoder_head,
                                   mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size,
                                   decoder_emb_dim,
                                   decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        pred_img, mask, alpha, beta = self.decoder(features,  backward_indexes)
        
        return pred_img, mask, alpha, beta


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat(
                    [self.cls_token.expand(-1, patches.shape[1], -1), patches],
                     dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        
        logits = self.head(features[0])

        return logits
    

class ViT_Segmentor(torch.nn.Module):
    def __init__(self,
                 encoder: MAE_Encoder,
                 image_size=224,
                 patch_size=16,
                 encoder_emb_dim=1024,
                 num_classes=2,
                 feature_layers=[5, 11, 17, 23]) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm

        # image paramters
        self.image_size = image_size
        self.patch_size = patch_size

        # UNETR decoder parameters
        self.feature_layers = sorted(feature_layers)
        self.num_layers = len(self.transformer)
        
        # decoder configure
        self.decoders = torch.nn.ModuleList([
            None,
            DecoderStage(512, 256),
            DecoderStage(256, 128),
            DecoderStage(128, 64)
        ])
        
        # skip connection process
        self.skip_conv = torch.nn.ModuleList([
            torch.nn.Conv2d(encoder_emb_dim, 512, 1),
            torch.nn.Conv2d(encoder_emb_dim, 256, 1),
            torch.nn.Conv2d(encoder_emb_dim, 128, 1),
            torch.nn.Conv2d(encoder_emb_dim, 64, 1)
        ])
        
        # final output layers
        self.final_conv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        x = rearrange(patches, 't b c -> b t c')

        # collecting multi-layer features
        features = []
        for i, blk in enumerate(self.transformer):
            x = blk(x)
            if i in self.feature_layers:
                spatial_feat = rearrange(x[:, 1:], 'b (h w) c -> b c h w', 
                                         h=self.image_size//self.patch_size,
                                         w=self.image_size//self.patch_size)
                features.append(spatial_feat)
                
        # adding final layer feature
        x = self.layer_norm(x)
        final_feat = rearrange(x[:, 1:], 'b (h w) c -> b c h w', 
                               h=self.image_size//self.patch_size,
                               w=self.image_size//self.patch_size).contiguous()
        features.append(final_feat)
        
        # reverse features (from deep to shallow)
        features = features[::-1]

        # UNETR decode processing
        x = None
        for i, (decoder, skip_conv) in enumerate(zip(self.decoders, self.skip_conv)):
            skip = skip_conv(features[i])
            skip = F.interpolate(skip, scale_factor=2**i, 
                               mode='bilinear', align_corners=True)

            if x is None:  
                x = skip # initial feature
            else:
                x = decoder(x)
                x = x + skip # feature fusion
            
        # final output
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        return x
    

class DecoderStage(torch.nn.Module):
    # UNETR decoder stage
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)