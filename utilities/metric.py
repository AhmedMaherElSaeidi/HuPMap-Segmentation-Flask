import numpy as np


# calculate IoU, and Dice
def calculate_metrics(y_true, y_pred):
    # True Positives, False Positives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Dice coefficient
    dice_denominator = 2 * tp + fp + fn
    dice = (2 * tp) / dice_denominator if dice_denominator != 0 else 1

    # Intersection over Union (IoU)
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator != 0 else 1

    return iou, dice
