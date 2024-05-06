# HuBMAP - Hacking the Human Vasculature (BackEnd implementation using Flask)

## Description
The goal is to segment instances of microvascular structures, including capillaries, arterioles, and venules, to automate the segmentation of microvasculature structures to improve researchers' understanding of how the blood vessels are arranged in human tissues.

## APIs
### `/predict/unet`
- **Description:** API for predicting using the UNet model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto the image.

### `/predict/linknet`
- **Description:** API for predicting using LinkNet model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto the image.

### `/predict/fcn`
- **Description:** API for predicting using FCN model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto the image.

### `/predict/ensemble`
- **Description:** API for predicting using an ensemble of models.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto the image.

## Installation
To install the project dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This command will install all the necessary packages listed in the `requirements.txt` file.


This command will install all the necessary packages listed in the `requirements.txt` file.

### Additional Setup

To ensure the project runs smoothly, please follow these steps:

1. **Weights Directory**: Create a directory named `weights` within the model directory. This directory should contain the weights of the three models (UNet, LinkNet, FCN). These weights are essential for the proper functioning of the project. Adjust the model weights file name to match the one in the `model_weights.json` file.

    - **UNet Model Weights**: Download the UNet model weights from [here](http://example.com/unet_weights).
    - **LinkNet Model Weights**: Download the LinkNet model weights from [here](http://example.com/linknet_weights).
    - **FCN Model Weights**: Download the FCN model weights from [here](http://example.com/fcn_weights).

2. **Images Directory in Static**: Create a directory named `images` within the `static` directory. This directory will contain the images generated by the server to display them later.

Make sure to create these directories and add the necessary files before running the project.


