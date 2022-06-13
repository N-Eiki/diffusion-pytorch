import os

from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout
    )
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as transforms
import cv2
from PIL import Image


def get_transforms(use_transform):
    if use_transform=="albmentations":
        return Compose([
            Resize(32, 32),
            # RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
            HorizontalFlip(p=0.5),
            # RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
            # HueSaturationValue(p=0.2, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            # ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
            # CoarseDropout(p=0.2),
            # Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    # if use_transform=="torchvision":
    #     return transforms.Compose([
    #         transforms.Resize(32, 32),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #         ),
    #         transforms.ToTensor(),
    #     ])

class DiffusionDataset(Dataset):
    def __init__(self, paths, use_transform="albmentations"):
        self.paths = paths

        self.transform = get_transforms(use_transform)

    def __len__(self, ):
        return len(self.paths)

    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image = augmented['image']
        return image
