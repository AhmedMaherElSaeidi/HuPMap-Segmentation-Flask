import os
import numpy as np
from utilities.utility import get_model_weights

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


class LinknetSegmentationModel:
    def __init__(self, backbone='efficientnetb5', weights_path=get_model_weights('linknet')):
        self.BACKBONE = backbone
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        self.model = sm.Linknet(self.BACKBONE, encoder_weights='imagenet')

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)

    def preprocess_image(self, image):
        preprocessed_image = self.preprocess_input(image)
        return preprocessed_image

    def predict(self, image, threshold=None):
        # Preprocess the input image
        preprocessed_image = self.preprocess_image(image)

        # Perform prediction using the loaded model
        predictions = self.model.predict(np.expand_dims(preprocessed_image, axis=0))

        # Process predictions as needed
        if threshold:
            predictions = (predictions > threshold).astype(np.uint8)

        return predictions
