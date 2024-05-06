import numpy as np


# calculate IoU from predicted prob
def iou_score(y_true, y_hat):
    score_list = []

    for k in range(y_hat.shape[0]):

        and_score = np.sum(y_hat[k][y_true[k] == 1])
        or_score = np.sum(y_true[k]) + np.sum(y_hat[k]) - and_score

        if or_score == 0:
            score = 1
        else:
            score = and_score / or_score
        score_list.append(score)

    return np.round(np.mean(np.array(score_list)), 5)


# calculate Dice from predicted prob
def dice_score(y_true, y_hat):
    dice_list = []

    for k in range(y_hat.shape[0]):
        TP = np.sum((y_true[k] == 1) & (y_hat[k] == 1))
        FP = np.sum((y_true[k] == 0) & (y_hat[k] == 1))
        FN = np.sum((y_true[k] == 1) & (y_hat[k] == 0))

        denominator = 2 * TP + FP + FN

        # Handle the case when the denominator is zero
        if denominator == 0:
            dice = 1  # Return 1, indicating perfect agreement
        else:
            # Calculate Dice coefficient
            dice = (2 * TP) / denominator

        dice_list.append(dice)

    average_dice = np.mean(dice_list)

    return np.round(average_dice, 5)
