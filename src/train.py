# src/train.py
"""
Training loop, validation, and checkpointing for Mask2Former traffic image segmentation.
"""


import os
import torch
import logging
from torch.optim import AdamW
from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, CHECKPOINT_DIR, ADE_MEAN, ADE_STD
from src.dataset import get_dataloader
from src.model.mask2former import CustomMask2Former
from src.utils import set_seed, save_checkpoint


def train_model(train_images, train_masks, val_images, val_masks):
    """
    Trains Mask2Former for semantic segmentation, logs loss, and saves checkpoints.
    Handles errors robustly.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    set_seed()
    try:
        model = CustomMask2Former().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = get_dataloader(train_images, train_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=True, is_train=True)
        val_loader = get_dataloader(val_images, val_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=False, is_train=False)
    except Exception as e:
        logging.error(f"Failed to initialize training: {e}")
        raise RuntimeError(f"Failed to initialize training: {e}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            try:
                images = batch.transformed_image.to(DEVICE)
                masks = batch.transformed_segmentation_map.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                logging.error(f"Error during training batch: {e}")
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('nan')
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_checkpoint({'epoch': epoch+1, 'model_state_dict': model.state_dict()},
                        os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
        # Optionally: add validation, metrics, and early stopping here
    return model
