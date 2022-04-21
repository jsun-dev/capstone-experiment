# USAGE
# python predict.py

# Import necessary packages
from src import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def prepare_plot(origImage, origMask, predMask):
    # Initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # Plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # Set the layout of the figure and display it
    figure.tight_layout()
    figure.show()
    figure.waitforbuttonpress()


def make_predictions(model, imagePath):
    # Set model to evaluation mode
    model.eval()

    # Turn off gradient tracking
    with torch.no_grad():
        # Load the image from disk, swap its color channels, cast it
        # to float data type, and scale its pixel values
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0

        # Resize the image and make a copy of it for visualization
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # Find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH, filename)

        # Load the ground-truth segmentation mask in grayscale mode and resize it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
                                     config.INPUT_IMAGE_HEIGHT))

        # Make the channel axis to be the leading one, add a batch
        # dimension, create a PyTorch tensor, and flash it to the
        # current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # Make the prediction, pass the results through the sigmoid
        # function, and convert the result to a NumPy array
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        # Filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        # Prepare a plot for visualization
        prepare_plot(orig, gtMask, predMask)


# Load the image paths in our testing file and randomly select 10 image paths
print("[INFO] Loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# Load our model from disk and flash it to the current device
print("[INFO] Load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# Iterate over the randomly selected test image paths
for path in imagePaths:
    # Make predictions and visualize the results
    make_predictions(unet, path)
