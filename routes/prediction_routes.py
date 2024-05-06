from utility import overlay_mask, save_images
from model.unet import UnetSegmentationModel
from flask import request, jsonify
from flask import Blueprint
import numpy as np
import uuid
import cv2

prediction_route = Blueprint("prediction_route", __name__)


@prediction_route.route('/')
def index():
    return 'Welcome to my Flask App!'


@prediction_route.route('/predict/unet', methods=['POST'])
def predict_unet():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return 'No file uploaded', 400

    # Get image from request header
    image_file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '':
        return 'No selected file', 400

    # If the file is provided, and it has the allowed file extension
    if image_file and image_file.filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.tif')):
        # Declaring static path
        static_path = "static/images/{}".format(uuid.uuid4())

        # Read the image file
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Check if image shape is (512, 512, 3)
        if image.shape != (512, 512, 3):
            return 'Image shape must be (512, 512, 3)', 400

        # Create an instance of the UnetSegmentationModel class
        model = UnetSegmentationModel()
        prediction = model.predict(image)[0]

        # Save the image, mask, and overlay
        image_paths = save_images(image, prediction, overlay_mask(image, prediction), static_path)

        # Return the prediction as JSON
        return jsonify(image_paths)
    else:
        return 'Invalid file format. Please upload an image (TIF, JPEG, JPG, PNG, or GIF)', 400