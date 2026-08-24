# evaluate_model.py
"""
Script to evaluate Mask2Former on traffic image segmentation data.

Loads the latest checkpoint from CHECKPOINT_DIR (written per-epoch by src.train.train_model) if
one exists. Without a checkpoint, this evaluates the freshly pretrained-but-not-finetuned model
(random class head) and logs a warning, rather than silently reporting meaningless metrics as
though they came from a trained model.
"""
import os
import glob
import re
import logging
import torch
from src.evaluate import evaluate_model
from src.model.mask2former import CustomMask2Former
from src.config import PROCESSED_DATA_DIR, BATCH_SIZE, DEVICE, ADE_MEAN, ADE_STD, CHECKPOINT_DIR
from src.dataset import get_dataloader


def _latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None

    def epoch_num(path):
        match = re.search(r"checkpoint_epoch_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=epoch_num)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        test_images = os.path.join(PROCESSED_DATA_DIR, "test", "images")
        test_masks = os.path.join(PROCESSED_DATA_DIR, "test", "masks")
        if not os.path.isdir(test_images) or not os.path.isdir(test_masks):
            raise FileNotFoundError(f"Test images or masks directory not found: {test_images}, {test_masks}")

        model = CustomMask2Former().to(DEVICE)

        checkpoint_path = _latest_checkpoint(CHECKPOINT_DIR)
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
