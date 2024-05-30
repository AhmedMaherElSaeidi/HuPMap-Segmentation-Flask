from utilities.image_handler import save_images
from utilities.metric import calculate_metrics
from model.linknet import LinknetSegmentationModel
from model.unet import UnetSegmentationModel
from model.unet_scratch import UNet
from model.fcn import FCN
from flask import Blueprint
from flask import request
import numpy as np
import uuid
import cv2

prediction_route = Blueprint("prediction_route", __name__)


@prediction_route.route('/')
def index():
    return 'Welcome to my Flask App!'


@prediction_route.route('/predict/unet', methods=['POST'])
def predict_unet():
    # Check if both "image" and "mask" files were uploaded
    if 'image' not in request.files or 'mask' not in request.files:
        return 'Both image and mask files are required', 400

    # Get image and mask files from request
    image_file = request.files['image']
    mask_file = request.files['mask']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '' or mask_file.filename == '':
        return 'Both image and mask files must be selected', 400

    # If the files are provided, and they have the allowed file extensions
    if image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')) and \
            mask_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Read the mask file
        mask_bytes = mask_file.read()
        np_mask = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(np_mask, cv2.IMREAD_GRAYSCALE)

        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            return 'Image shape must be (512, 512, 3) and mask shape must be (512, 512)', 400

        # Add extra dimension to mask if it's not (512, 512, 1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # time to make a prediction
        threshold = 0.5
        unet = UnetSegmentationModel()
        prediction = unet.predict(image, threshold=threshold)[0]

        # Set the response dictionary
        response = save_images(static_path, image, mask, prediction)
        metrics = calculate_metrics(mask, prediction)
        response.update({
            "iou_score": metrics[0],
            "dice_score": metrics[1],
            "threshold": threshold * 100
        })

        return response, 201
    else:
        return 'Invalid file format. Please upload images in (TIF, JPEG, JPG, PNG, or GIF) format', 400


@prediction_route.route('/predict/linknet', methods=['POST'])
def predict_linknet():
    # Check if both "image" and "mask" files were uploaded
    if 'image' not in request.files or 'mask' not in request.files:
        return 'Both image and mask files are required', 400

    # Get image and mask files from request
    image_file = request.files['image']
    mask_file = request.files['mask']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '' or mask_file.filename == '':
        return 'Both image and mask files must be selected', 400

    # If the files are provided, and they have the allowed file extensions
    if image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')) and \
            mask_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Read the mask file
        mask_bytes = mask_file.read()
        np_mask = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(np_mask, cv2.IMREAD_GRAYSCALE)

        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            return 'Image shape must be (512, 512, 3) and mask shape must be (512, 512)', 400

        # Add extra dimension to mask if it's not (512, 512, 1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # time to make a prediction
        threshold = 0.5
        unet = LinknetSegmentationModel()
        prediction = unet.predict(image, threshold=threshold)[0]

        # Make the mask binary for the next stages
        binary_mask = np.where(mask == 255, 1, 0)
        binary_mask = binary_mask.astype(np.uint8)

        # Set the response dictionary
        response = save_images(static_path, image, binary_mask, prediction)
        metrics = calculate_metrics(binary_mask, prediction)
        response.update({
            "iou_score": metrics[0],
            "dice_score": metrics[1],
            "threshold": threshold * 100
        })

        return response, 201
    else:
        return 'Invalid file format. Please upload images in (TIF, JPEG, JPG, PNG, or GIF) format', 400


@prediction_route.route('/predict/fcn', methods=['POST'])
def predict_fcn():
    # Check if both "image" and "mask" files were uploaded
    if 'image' not in request.files or 'mask' not in request.files:
        return 'Both image and mask files are required', 400

    # Get image and mask files from request
    image_file = request.files['image']
    mask_file = request.files['mask']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '' or mask_file.filename == '':
        return 'Both image and mask files must be selected', 400

    # If the files are provided, and they have the allowed file extensions
    if image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')) and \
            mask_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Read the mask file
        mask_bytes = mask_file.read()
        np_mask = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(np_mask, cv2.IMREAD_GRAYSCALE)

        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            return 'Image shape must be (512, 512, 3) and mask shape must be (512, 512)', 400

        # Add extra dimension to mask if it's not (512, 512, 1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # time to make a prediction
        threshold = 0.5
        fcn = FCN()
        prediction = fcn.predict(image, threshold=threshold)[0]

        # Set the response dictionary
        response = save_images(static_path, image, mask, prediction)
        metrics = calculate_metrics(mask, prediction)
        response.update({
            "iou_score": metrics[0],
            "dice_score": metrics[1],
            "threshold": threshold * 100
        })

        return response, 201
    else:
        return 'Invalid file format. Please upload images in (TIF, JPEG, JPG, PNG, or GIF) format', 400


@prediction_route.route('/predict/unet_scratch', methods=['POST'])
def predict_unet_scratch():
    # Check if both "image" and "mask" files were uploaded
    if 'image' not in request.files or 'mask' not in request.files:
        return 'Both image and mask files are required', 400

    # Get image and mask files from request
    image_file = request.files['image']
    mask_file = request.files['mask']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '' or mask_file.filename == '':
        return 'Both image and mask files must be selected', 400

    # If the files are provided, and they have the allowed file extensions
    if image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')) and \
            mask_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Read the mask file
        mask_bytes = mask_file.read()
        np_mask = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(np_mask, cv2.IMREAD_GRAYSCALE)

        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            return 'Image shape must be (512, 512, 3) and mask shape must be (512, 512)', 400

        # Add extra dimension to mask if it's not (512, 512, 1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # time to make a prediction
        threshold = 0.5
        unet = UNet()
        prediction = unet.predict(image, threshold=threshold)[0]

        # Set the response dictionary
        response = save_images(static_path, image, mask, prediction)
        metrics = calculate_metrics(mask, prediction)
        response.update({
            "iou_score": metrics[0],
            "dice_score": metrics[1],
            "threshold": threshold * 100
        })

        return response, 201
    else:
        return 'Invalid file format. Please upload images in (TIF, JPEG, JPG, PNG, or GIF) format', 400


@prediction_route.route('/predict/ensemble', methods=['POST'])
def predict_ensemble():
    # Check if both "image" and "mask" files were uploaded
    if 'image' not in request.files or 'mask' not in request.files:
        return 'Both image and mask files are required', 400

    # Get image and mask files from request
    image_file = request.files['image']
    mask_file = request.files['mask']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '' or mask_file.filename == '':
        return 'Both image and mask files must be selected', 400

    # If the files are provided, and they have the allowed file extensions
    if image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')) and \
            mask_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Read the mask file
        mask_bytes = mask_file.read()
        np_mask = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(np_mask, cv2.IMREAD_GRAYSCALE)

        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            return 'Image shape must be (512, 512, 3) and mask shape must be (512, 512)', 400

        # Add extra dimension to mask if it's not (512, 512, 1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # # time to make a prediction
        # threshold = 0.5
        # ...
        #
        # # Set the response dictionary
        # response = save_images(static_path, image, mask, prediction)
        # metrics = calculate_metrics(mask, prediction)
        # response.update({
        #     "iou_score": metrics[0],
        #     "dice_score": metrics[1],
        #     "threshold": threshold*100
        # })

        return "under maintenance", 201
    else:
        return 'Invalid file format. Please upload images in (TIF, JPEG, JPG, PNG, or GIF) format', 400
