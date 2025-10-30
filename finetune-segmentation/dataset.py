import os
import torch
import numpy as np
from torchvision.transforms import v2

from PIL import Image
from torch.utils.data import Dataset


class DualSegmentationTransform:
    def __init__(self):
        self.angle_range = (-15, 15)
        self.flip_prob = 0.5

    def __call__(self, img, mask):
        # random rotation
        if torch.rand(1) > 0.5:
            angle = torch.empty(1).uniform_(*self.angle_range).item()
            img = v2.functional.rotate(img, angle)
            mask = v2.functional.rotate(mask, angle)
        
        # random horizontal flip
        if torch.rand(1) > self.flip_prob:
            img = v2.functional.hflip(img)
            mask = v2.functional.hflip(mask)
            
        return img, mask
    

class SegmentationDataset(Dataset):
    def __init__(self, root_path, input_size, is_dual_transform=False):
        self.image_paths = root_path + '/images'
        self.mask_paths = root_path + '/masks'
        self.is_dual_transform = is_dual_transform
        self.transform = DualSegmentationTransform()

        self.image_list = sorted(os.listdir(self.image_paths))
        self.mask_list = sorted(os.listdir(self.mask_paths))

        self.all_image_paths = [os.path.join(self.image_paths, file) for file in self.image_list]
        self.all_mask_paths = [os.path.join(self.mask_paths, file) for file in self.mask_list]
        
        if len(self.all_image_paths) != len(self.all_mask_paths):
            raise ValueError("The number of images is not equal to that of masks!")

        # image preprocess
        self.img_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((input_size, input_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = v2.Compose([
            v2.Resize((input_size, input_size), interpolation=v2.InterpolationMode.NEAREST),
            v2.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
        ])

        # dynamically determine the number of categories
        self.num_class = self._determine_num_classes()

    def _determine_num_classes(self):
        # scan all mask files and determine the total number of unique grayscale values
        unique_values = set()
        for mask_path in self.all_mask_paths:
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            unique_values.update(np.unique(mask_array))
        num_classes = len(unique_values)
        print(f"Detected {num_classes} unique gray values in ground-truth mask: {sorted(unique_values)}")

        return num_classes

    def map_mask_values(self, mask):
        # dynamically map grayscale values ​​to continuous category indices [0, 1, 2, ...]
        unique_values = torch.unique(mask)
        mapping = {val.item(): idx for idx, val in enumerate(sorted(unique_values))}
        mapped_mask = torch.zeros_like(mask, dtype=torch.long)
        for gray_value, class_idx in mapping.items():
            mapped_mask[mask == gray_value] = class_idx

        return mapped_mask
    
    def __len__(self):
        return len(self.all_image_paths)
    
    def __getitem__(self, idx):
        # Check if image and mask filenames match (excluding extensions)
        image_filename = os.path.splitext(os.path.basename(self.all_image_paths[idx]))[0]
        mask_filename = os.path.splitext(os.path.basename(self.all_mask_paths[idx]))[0]
        if image_filename != mask_filename:
            raise ValueError(f"Filename mismatch: image '{image_filename}' vs mask '{mask_filename}' at index {idx}")
    
        image = Image.open(self.all_image_paths[idx]).convert('RGB')
        mask = Image.open(self.all_mask_paths[idx]).convert('L')
        img_path = self.all_image_paths[idx]

        # if image.size != mask.size:
            # print(img_path)
            # raise ValueError("Size mismatch: image {} vs mask {} [image path: {}]".format(image.size, mask.size, img_path))
        
        if self.is_dual_transform:
            image, mask = self.transform(image, mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = self.map_mask_values(mask)

        return image, mask, self.mask_list[idx]