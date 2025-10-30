import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DualEnhancementTransform:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, img1, img2):
        if torch.rand(1) < self.flip_prob:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)

        if torch.rand(1) < self.rotate_prob:
            angle = torch.randint(1, 4, (1,)).item() * 90
            img1 = transforms.functional.rotate(img1, angle)
            img2 = transforms.functional.rotate(img2, angle)

        return img1, img2


class EnhancementDataset(Dataset):
    def __init__(self, root_path, input_size, is_dual_transform=False):
        self.low_quality_path = root_path + '/low_quality'
        self.high_quality_path = root_path + '/high_quality'
        self.is_dual_transform = is_dual_transform
        self.dual_transform = DualEnhancementTransform()

        self.low_name_list = sorted(os.listdir(self.low_quality_path))
        self.high_name_list = sorted(os.listdir(self.high_quality_path))

        self.all_low_paths = [os.path.join(self.low_quality_path, file_name) for file_name in self.low_name_list]
        self.all_high_paths = [os.path.join(self.high_quality_path, file_name) for file_name in self.high_name_list]
        
        if len(self.all_low_paths) != len(self.all_high_paths):
            raise ValueError("The number of low quality images is not equal to that of high quality images!")

        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.all_low_paths)
    
    def __getitem__(self, idx):
        # print(self.all_low_paths[idx])
        # print(self.all_high_paths[idx])
        low_image = Image.open(self.all_low_paths[idx]).convert('RGB')
        high_image = Image.open(self.all_high_paths[idx]).convert('RGB')
        low_path = self.all_low_paths[idx]

        if low_image.size != high_image.size:
            raise ValueError("Size mismatch: low quality image {} vs high quality image {} [low quality path: {}]".format(low_image.size, high_image.size, low_path))
        
        if self.is_dual_transform:
            low_image, high_image = self.dual_transform(low_image, high_image)
        
        low_image = self.img_transform(low_image)
        high_image = self.img_transform(high_image)
    
        return low_image, high_image, self.high_name_list[idx]