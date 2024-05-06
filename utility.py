from globals import HOST, PORT
import numpy as np
import json
import cv2
import os


def overlay_mask(image, mask, opacity=0.5):
    if np.max(mask) == 0:
        return image.astype(np.uint8)  # Return the original image if the mask is blank (all zeros)

    mask_normalized = mask.astype(float) / np.max(mask)
    alpha = mask_normalized[:, :, 0] * opacity  # Extract the single channel from the mask & Adjust the opacity by
    # multiplying with a factor
    alpha = alpha[:, :, np.newaxis]  # Add a third dimension to make it compatible with the image
    result = alpha * mask + (1 - alpha) * image

    return result.astype(np.uint8)


def save_images(image, mask, overlay, static_path):
    # Save the original image
    biomedical_image_path = static_path + "_image.png"
    cv2.imwrite(biomedical_image_path, image)

    # Save the mask predicted by model
    biomedical_mask_path = static_path + "_mask.png"
    cv2.imwrite(biomedical_mask_path, mask * 255)

    # Save the overlay mask image
    biomedical_overlay_path = static_path + "_overlay.png"
    cv2.imwrite(biomedical_overlay_path, overlay_mask(image, mask))

    return dict(image="http://{}:{}/{}".format(HOST, PORT, biomedical_image_path),
                mask="http://{}:{}/{}".format(HOST, PORT, biomedical_mask_path),
                overlay="http://{}:{}/{}".format(HOST, PORT, biomedical_overlay_path))


def get_model_path(model_name):
    json_file_path = "model_weights.json"

    # Read the JSON file
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    # Access model paths
    model_weights_path = data["models"][model_name]

    return model_weights_path


def prone_static_dir(folder_path):
    try:
        # Get the list of files and directories in the folder
        contents = os.listdir(folder_path)

        # Iterate over the contents and remove each one
        for item in contents:
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                os.rmdir(item_path)

        print("Contents of '{}' deleted successfully.".format(folder_path))
    except Exception as e:
        print(f"An error occurred: {e}")



