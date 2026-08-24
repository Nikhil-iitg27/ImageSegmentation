# src/train.py
"""
Training loop, validation, and checkpointing for Mask2Former traffic image segmentation.
Mirrors the training procedure in notebooks/Project.ipynb (cell 67): uses the model's own
built-in matching loss over mask_labels/class_labels (not per-pixel cross-entropy on a flat
logits tensor), and validates each epoch on a fixed subset of the validation set.
"""

import os
import logging
from torch.optim import Adam
from src.config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DEVICE, CHECKPOINT_DIR, ADE_MEAN, ADE_STD
from src.dataset import get_dataloader
from src.model.mask2former import CustomMask2Former
from src.evaluate import evaluate_model
from src.utils import set_seed, save_checkpoint


def train_model(
    train_images,
    train_masks,
    val_images,
    val_masks,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    log_interval=100,
    val_max_batches=6,
):
    """
    Trains Mask2Former for semantic segmentation, logs loss, validates each epoch, and saves
    checkpoints. val_max_batches=6 matches notebook cell 67's per-epoch validation subset size
    (the full validation set is only used for the final post-training evaluation, mirroring
    cell 70 in evaluate_model.py).
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    set_seed()
    try:
        model = CustomMask2Former().to(DEVICE)
        # freeze_pixel_level_module() already ran inside CustomMask2Former.__init__, so
        # optimizer.parameters() below includes frozen params, but they never receive/apply
        # gradients (matches notebook cell 67, which also passes model.parameters() unfiltered).
        optimizer = Adam(model.parameters(), lr=learning_rate)
        train_loader = get_dataloader(train_images, train_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=True, is_train=True)
        val_loader = get_dataloader(val_images, val_masks, BATCH_SIZE, ADE_MEAN, ADE_STD, shuffle=False, is_train=False)
    except Exception as e:
        logging.error(f"Failed to initialize training: {e}")
        raise RuntimeError(f"Failed to initialize training: {e}")

    loss_history = []
    val_metric_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0
        for idx, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                outputs = model(
                    pixel_values=batch["pixel_values"].to(DEVICE),
                    mask_labels=[labels.to(DEVICE) for labels in batch["mask_labels"]],
                    class_labels=[labels.to(DEVICE) for labels in batch["class_labels"]],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                batch_size = batch["pixel_values"].size(0)
                running_loss += loss.item()
                num_samples += batch_size

                if idx % log_interval == 0 and idx > 0:
                    logging.info(f"Iteration {idx} - loss: {running_loss / num_samples}")
            except Exception as e:
                logging.error(f"Error during training batch {idx}: {e}")

        epoch_loss = running_loss / num_samples if num_samples > 0 else float('nan')
        loss_history.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        val_metrics = evaluate_model(model, val_loader, max_batches=val_max_batches)
        val_metric_history.append(val_metrics)
        logging.info(
            f"Validation Metrics - Mean Dice: {val_metrics['mean_dice']:.4f}, "
            f"Mean F1 (β=0.5): {val_metrics['mean_f1']:.4f}, Mean IoU: {val_metrics['mean_iou']:.4f}"
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        save_checkpoint(
            {'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'loss': epoch_loss},
            os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch + 1}.pth'),
        )

    return model, {'loss': loss_history, 'val_metrics': val_metric_history}
