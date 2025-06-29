# src/evaluate.py
"""
Evaluation metrics and visualization for Mask2Former traffic image segmentation.
"""



import torch
import numpy as np
import logging
from src.utils import plot_sample
from src.config import ADE_MEAN, ADE_STD

def compute_iou(pred_mask, true_mask, num_classes):
    ious = []
    pred_mask = pred_mask.cpu().numpy().flatten()
    true_mask = true_mask.cpu().numpy().flatten()
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        target_inds = true_mask == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

def evaluate_model(model, dataloader, num_classes, device):
    """
    Evaluates the model on a dataloader and prints mean IoU. Visualizes predictions.
    Handles errors robustly.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for batch in dataloader:
            try:
                images = batch.transformed_image.to(device)
                masks = batch.transformed_segmentation_map.to(device)
                outputs = model(images).logits
                preds = torch.argmax(outputs, dim=1)
                for i in range(images.shape[0]):
                    iou = compute_iou(preds[i], masks[i], num_classes)
                    iou_scores.append(iou)
                    plot_sample(batch.original_image[i], masks[i].cpu(), preds[i].cpu())
            except Exception as e:
                logging.error(f"Error during evaluation batch: {e}")
    mean_iou = np.nanmean(iou_scores) if iou_scores else float('nan')
    logging.info(f"Mean IoU: {mean_iou:.4f}")
    return mean_iou
