# HuBMAP - Hacking the Human Vasculature (BackEnd implementation using flask)

## Description
The goal is to segment instances of microvascular structures, including capillaries, arterioles, and venules, to in automating the segmentation of microvasculature structures as it will improve researchers' understanding of how the blood vessels are arranged in human tissues.

## APIs
### /predict/unet
- **Description:** API for predicting using UNet model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto image.

### /predict/linknet
- **Description:** API for predicting using LinkNet model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto image.

### /predict/fcn
- **Description:** API for predicting using FCN model.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto image.

### /predict/ensemble
- **Description:** API for predicting using an ensemble of models.
- **Input:** Kidney tissue image of shape `512x512x3`.
- **Output:**
  - Path of the predicted mask.
  - Path of the kidney tissue image received.
  - Path of the mask overlaid onto image.

## Installation
To install the project dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This command will install all the necessary packages listed in the `requirements.txt` file.

