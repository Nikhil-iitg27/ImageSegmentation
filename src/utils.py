# src/utils.py
"""
Utility functions for logging, visualization, and reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets random seed for reproducibility across numpy, random, and torch.
    Logs the seed used.
    """
    import logging
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def save_checkpoint(state, filename):
    """
    Saves model state to a file and logs the action.
    """
    import logging
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def plot_sample(image, mask, pred_mask=None, class_names=None):
    """
    Visualizes an image, its ground truth mask, and optionally a predicted mask.
    Handles both numpy arrays and torch tensors robustly. Logs errors if visualization fails.
    """
    import logging
    try:
        import matplotlib.pyplot as plt
        n_plots = 3 if pred_mask is not None else 2
        plt.figure(figsize=(12, 4))
        plt.subplot(1, n_plots, 1)
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        else:
            img = image
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, n_plots, 2)
        plt.imshow(mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask, cmap='jet', alpha=0.7)
        plt.title('Ground Truth')
        plt.axis('off')
        if pred_mask is not None:
            plt.subplot(1, n_plots, 3)
            plt.imshow(pred_mask.cpu().numpy() if isinstance(pred_mask, torch.Tensor) else pred_mask, cmap='jet', alpha=0.7)
            plt.title('Prediction')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Plotting failed: {e}")
