# src/model/mask2former.py
"""
Mask2Former model definition and utilities for traffic image segmentation.
Uses Hugging Face transformers for Mask2Former backbone.
"""

import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation
from src.config import NUM_CLASSES

class CustomMask2Former(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained_model="facebook/mask2former-swin-large-coco-panoptic"): 
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model,
            id2label={i: str(i) for i in range(num_classes)},
            label2id={str(i): i for i in range(num_classes)},
            ignore_mismatched_sizes=True
        )
        self.model.config.num_labels = num_classes

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
