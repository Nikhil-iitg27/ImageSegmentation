# train_model.py
"""
Script to train Mask2Former on traffic image segmentation data.
"""

from src.train import train_model
from src.config import PROCESSED_DATA_DIR
import os

def main():
    try:
        train_images = os.path.join(PROCESSED_DATA_DIR, "train", "images")
        train_masks = os.path.join(PROCESSED_DATA_DIR, "train", "masks")
        val_images = os.path.join(PROCESSED_DATA_DIR, "val", "images")
        val_masks = os.path.join(PROCESSED_DATA_DIR, "val", "masks")
        for d in [train_images, train_masks, val_images, val_masks]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Required directory not found: {d}")
        model = train_model(train_images, train_masks, val_images, val_masks)
        print("Training complete.")
    except Exception as e:
        print(f"Error in train_model.py: {e}")

if __name__ == "__main__":
    main()
