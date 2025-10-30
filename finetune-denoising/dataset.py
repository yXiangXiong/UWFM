import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DualDenoisingTransform:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img1, img2):
        # Random horizontal flip
        if torch.rand(1) > self.flip_prob:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)

        return img1, img2


class DenoisingDataset(Dataset):
    def __init__(self, root_path, input_size, is_dual_transform=False):
        self.noised_path = root_path + '/noised'
        self.clean_path = root_path + '/clean'
        self.is_dual_transform = is_dual_transform
        self.dual_transform = DualDenoisingTransform()

        self.noised_name_list = sorted(os.listdir(self.noised_path))
        self.clean_name_list = sorted(os.listdir(self.clean_path))

        self.noised_paths = [os.path.join(self.noised_path, file_name) for file_name in self.noised_name_list]
        self.clean_paths = [os.path.join(self.clean_path, file_name) for file_name in self.clean_name_list]
        
        if len(self.noised_paths) != len(self.clean_paths):
            raise ValueError("The number of low noised images is not equal to that of clean images!")

        self.img_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.noised_paths)
    
    def __getitem__(self, idx):
        noised_image = Image.open(self.noised_paths[idx]).convert('RGB')
        clean_image = Image.open(self.clean_paths[idx]).convert('RGB')
        noised_path = self.noised_paths[idx]

        if noised_image.size != clean_image.size:
            raise ValueError("Size mismatch: low quality image {} vs high quality image {} [low quality path: {}]".format(noised_image.size, clean_image.size, noised_path))
        
        if self.is_dual_transform:
            noised_image, clean_image = self.dual_transform(noised_image, clean_image)
        
        noised_image = self.img_transform(noised_image)
        clean_image = self.img_transform(clean_image)
    
        return noised_image, clean_image, self.clean_name_list[idx]