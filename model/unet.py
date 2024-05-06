import os
import numpy as np
from utility import get_model_path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


class UnetSegmentationModel:
    def __init__(self, backbone='efficientnetb5', weights_path=get_model_path('unet')):
        self.BACKBONE = backbone
        self.preprocess_input = sm.get_preprocessing(self.BACKBONE)
        self.model = sm.Unet(self.BACKBONE, encoder_weights='imagenet')
        self.model.compile('AdamW', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])
        self.model.load_weights(weights_path)

    def preprocess_image(self, image):
        preprocessed_image = self.preprocess_input(image)
        return preprocessed_image

    def predict(self, image, threshold=0.5):
        # Preprocess the input image
        preprocessed_image = self.preprocess_image(image)

        # Perform prediction using the loaded model
        predictions = self.model.predict(np.expand_dims(preprocessed_image, axis=0))

        # Process predictions as needed
        predictions = (predictions > threshold).astype(np.uint8)

        return predictions
