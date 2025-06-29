# src/dataset.py
"""
Dataset loading, augmentation, and preprocessing for traffic image segmentation.
Follows the robust, modular style of the Mask2Former notebook implementation.
"""


import os
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class SegmentationDataInput:
    original_image: np.ndarray
    transformed_image: Any
    original_segmentation_map: np.ndarray
    transformed_segmentation_map: Any

class SegmentationTransform:
    def __init__(self, mean, std, is_train=True):
        self.is_train = is_train
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        if self.is_train:
            self.img_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                self.img_transform
            ])

    def __call__(self, image, mask):
        image = self.img_transform(image)
        mask = np.array(mask)
        return image, mask


class SemanticSegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation supporting both directory and in-memory (list) data sources.
    """
    def __init__(self, images, masks, mean, std, is_train=True, from_memory=False):
        self.from_memory = from_memory
        self.transform = SegmentationTransform(mean, std, is_train)
        try:
            if from_memory:
                assert len(images) == len(masks), "Number of images and masks must match."
                self.images = images
                self.masks = masks
            else:
                if not os.path.isdir(images) or not os.path.isdir(masks):
                    logging.error(f"Image or mask directory not found: {images}, {masks}")
                    raise FileNotFoundError(f"Image or mask directory not found: {images}, {masks}")
                self.images_dir = images
                self.masks_dir = masks
                self.images = sorted([f for f in os.listdir(images) if f.endswith(('.png', '.jpg', '.jpeg'))])
                self.masks = sorted([f for f in os.listdir(masks) if f.endswith(('.png', '.jpg', '.jpeg'))])
                if len(self.images) != len(self.masks):
                    logging.error(f"Number of images and masks must match. Found {len(self.images)} images and {len(self.masks)} masks.")
                assert len(self.images) == len(self.masks), "Number of images and masks must match."
        except Exception as e:
            logging.error(f"Failed to initialize SemanticSegmentationDataset: {e}")
            raise RuntimeError(f"Failed to initialize SemanticSegmentationDataset: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> SegmentationDataInput:
        try:
            if self.from_memory:
                image = self.images[idx]
                mask = self.masks[idx]
            else:
                img_path = os.path.join(self.images_dir, self.images[idx])
                mask_path = os.path.join(self.masks_dir, self.masks[idx])
                image = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path)
            transformed_image, transformed_mask = self.transform(image, mask)
            return SegmentationDataInput(
                original_image=np.array(image),
                transformed_image=transformed_image,
                original_segmentation_map=np.array(mask),
                transformed_segmentation_map=transformed_mask
            )
        except Exception as e:
            logging.error(f"Failed to load sample {idx}: {e}")
            raise RuntimeError(f"Failed to load sample {idx}: {e}")

def get_dataloader(images, masks, batch_size, mean, std, shuffle=True, is_train=True, from_memory=False):
    try:
        dataset = SemanticSegmentationDataset(images, masks, mean, std, is_train, from_memory=from_memory)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except Exception as e:
        logging.error(f"Failed to create dataloader: {e}")
        raise RuntimeError(f"Failed to create dataloader: {e}")
