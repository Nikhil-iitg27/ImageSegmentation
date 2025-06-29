# src/config.py
"""
Configuration file for Mask2Former Traffic Image Segmentation Project.
Contains paths, hyperparameters, and other global settings.
"""

import os
import torch

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model & output paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')


# Hyperparameters (update as needed)
NUM_CLASSES = 21  # Set according to your dataset
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
IMAGE_SIZE = (512, 512)
SEED = 42

# Normalization (ADE20K mean/std as in Mask2Former notebook)
ADE_MEAN = [123.675 / 255, 116.280 / 255, 103.530 / 255]
ADE_STD = [58.395 / 255, 57.120 / 255, 57.375 / 255]

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
