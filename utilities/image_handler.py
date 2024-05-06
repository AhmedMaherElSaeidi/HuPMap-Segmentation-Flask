import cv2
import numpy as np
from globals import HOST, PORT


def overlay_masks(y_true, y_hat, threshold=0.5):
    cutoff_img = (y_hat[:, :, 0] > threshold).astype(int)

    true_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    true_mask[:, :, 1] = y_true[:, :, 0] * 200

    pred_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    pred_mask[:, :, 2] = cutoff_img * 230

    overlay_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay_mask[:, :, 1] = y_true[:, :, 0] * 200  # Green for true mask
    overlay_mask[:, :, 2] = cutoff_img * 230  # Red for predicted mask

    return true_mask, pred_mask, overlay_mask


def save_images(static_path, kidney_image, true_mask, predicted_mask, threshold=0.5):
    # Save the kidney slide image
    biomedical_image_path = static_path + "_image.png"
    cv2.imwrite(biomedical_image_path, kidney_image)

    # Getting our RGB masks ready
    true_mask, predicted_mask, overlaid_mask = overlay_masks(true_mask, predicted_mask, threshold=threshold)

    # Save the true mask
    biomedical_true_mask_path = static_path + "_true_mask.png"
    cv2.imwrite(biomedical_true_mask_path, true_mask)

    # Save the predicted mask
    biomedical_predicted_mask_path = static_path + "_predicted_mask.png"
    cv2.imwrite(biomedical_predicted_mask_path, predicted_mask)

    # Save the overlaid masks
    biomedical_overlaid_mask_path = static_path + "_overlaid_mask.png"
    cv2.imwrite(biomedical_overlaid_mask_path, overlaid_mask)

    return dict(image="http://{}:{}/{}".format(HOST, PORT, biomedical_image_path),
                true_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_true_mask_path),
                predicted_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_predicted_mask_path),
                overlaid_mask="http://{}:{}/{}".format(HOST, PORT, biomedical_overlaid_mask_path))
