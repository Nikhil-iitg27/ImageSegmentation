# evaluate_model.py
"""
Script to evaluate Mask2Former on traffic image segmentation data.

Loads a checkpoint from CHECKPOINT_DIR (written by src.train.train_model) if one exists,
preferring checkpoint_best.pth (highest validation Dice seen during training) over
checkpoint_latest.pth (whatever epoch training last completed on -- only used as a fallback,
e.g. if best.pth wasn't written yet because training was interrupted before any epoch improved
on it). Without either, this evaluates the freshly pretrained-but-not-finetuned model (random
class head) and logs a warning, rather than silently reporting meaningless metrics as though
they came from a trained model.
"""
import os
import logging
import torch
from src.evaluate import evaluate_model
from src.model.mask2former import CustomMask2Former
from src.config import PROCESSED_DATA_DIR, BATCH_SIZE, DEVICE, ADE_MEAN, ADE_STD, CHECKPOINT_DIR
from src.dataset import get_dataloader
from src.train import LATEST_CHECKPOINT_NAME, BEST_CHECKPOINT_NAME


def _select_checkpoint(checkpoint_dir):
    best = os.path.join(checkpoint_dir, BEST_CHECKPOINT_NAME)
    if os.path.isfile(best):
        return best
    latest = os.path.join(checkpoint_dir, LATEST_CHECKPOINT_NAME)
    if os.path.isfile(latest):
        return latest
    return None


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        test_images = os.path.join(PROCESSED_DATA_DIR, "test", "images")
        test_masks = os.path.join(PROCESSED_DATA_DIR, "test", "masks")
        if not os.path.isdir(test_images) or not os.path.isdir(test_masks):
            raise FileNotFoundError(f"Test images or masks directory not found: {test_images}, {test_masks}")

        model = CustomMask2Former().to(DEVICE)

        checkpoint_path = _select_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            logging.info(f"Loaded checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
        else:
            logging.warning(
                f"No checkpoint found in {CHECKPOINT_DIR} — evaluating the pretrained-but-not-"
                "finetuned model. Run train_model.py first for meaningful metrics."
            )

        # batch_size=BATCH_SIZE (not the previous hardcoded 1) matches the notebook's
        # test_dataloader batch size; the Dice/F1/IoU metric is averaged per-sample regardless of
        # batch size, so this only affects evaluation speed, not the result.
        dataloader = get_dataloader(test_images, test_masks, batch_size=BATCH_SIZE, mean=ADE_MEAN, std=ADE_STD, shuffle=False, is_train=False)
        metrics = evaluate_model(model, dataloader)
        print(f"Test Metrics - Mean Dice: {metrics['mean_dice']:.4f}, Mean F1 (beta=0.5): {metrics['mean_f1']:.4f}, Mean IoU: {metrics['mean_iou']:.4f}")
    except Exception as e:
        print(f"Error in evaluate_model.py: {e}")

if __name__ == "__main__":
    main()
