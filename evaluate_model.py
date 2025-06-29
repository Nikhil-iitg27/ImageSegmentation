# evaluate_model.py
"""
Script to evaluate Mask2Former on traffic image segmentation data.
"""
from src.evaluate import evaluate_model
from src.model.mask2former import CustomMask2Former
from src.config import PROCESSED_DATA_DIR, NUM_CLASSES, DEVICE, ADE_MEAN, ADE_STD
from src.dataset import get_dataloader
import os

def main():
    try:
        test_images = os.path.join(PROCESSED_DATA_DIR, "test", "images")
        test_masks = os.path.join(PROCESSED_DATA_DIR, "test", "masks")
        if not os.path.isdir(test_images) or not os.path.isdir(test_masks):
            raise FileNotFoundError(f"Test images or masks directory not found: {test_images}, {test_masks}")
        model = CustomMask2Former().to(DEVICE)
        # Optionally: load checkpoint here
        dataloader = get_dataloader(test_images, test_masks, batch_size=1, mean=ADE_MEAN, std=ADE_STD, shuffle=False, is_train=False)
        evaluate_model(model, dataloader, NUM_CLASSES, DEVICE)
    except Exception as e:
        print(f"Error in evaluate_model.py: {e}")

if __name__ == "__main__":
    main()
