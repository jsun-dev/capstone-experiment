# Import necessary packages
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import imutils
import math

def prepare_plot(origImage, origMask, predMask, imgClosing, imgContours):
    # Initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))

    # Plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    ax[3].imshow(imgClosing)
    ax[4].imshow(imgContours)

    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    ax[3].set_title("Morphology")
    ax[4].set_title("Simplified Contour")

    # Set the layout of the figure and display it
    # figure.tight_layout()
    # figure.show()
    # figure.waitforbuttonpress()

    cv2.imshow('original', origImage)
    cv2.imshow('originalMask', origMask)
    cv2.imshow('predictedMask', predMask)
    cv2.imshow('morphology', imgClosing)
    # # cv2.imshow('canny', imgCanny)
    cv2.imshow('contours', imgContours)
    cv2.waitKey(0)


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

        # Make a copy of the image for visualization
        imageCopy = image.copy()

        # Resize the image
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))

        # Find the filename and generate the path to ground truth mask
        filename = imagePath.split(os.path.sep)[-1]
        groundTruthPath = os.path.join(config.MASK_DATASET_PATH, 'm_' + filename)

        # Load the ground-truth segmentation mask in grayscale mode and copy it
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMaskCopy = gtMask.copy()

        # Resize the ground truth mask
        gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT))

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

        # Resize the predicted mask
        predMask = cv2.resize(predMask, (gtMaskCopy.shape[1], gtMaskCopy.shape[0]))

        # Resize all images to a smaller dimension
        imageCopy = imutils.resize(imageCopy, height=1000)
        gtMaskCopy = imutils.resize(gtMaskCopy, height=1000)
        predMask = imutils.resize(predMask, height=1000)

        imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

        #######################
        # PROCESSING MASK
        #######################
        predMaskCopy = predMask.copy()

        # Apply morphological operations to erase unnecessary details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        imgOpening = cv2.morphologyEx(predMaskCopy, cv2.MORPH_OPEN, kernel, iterations=1)
        imgClosing = cv2.morphologyEx(imgOpening, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find all contours
        imgContours = np.zeros(imageCopy.shape, dtype=np.uint8)
        imgDocumentContour = imageCopy.copy()
        contours, hierarchy = cv2.findContours(imgClosing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            # # Draw all contours in blue
            # cv2.drawContours(imgContours, contours, -1, 255, 3)
            # Assume the largest contour is the document
            sortedContours = sorted(contours, key=cv2.contourArea)
            biggestContour = sortedContours[-1]
            x, y, w, h = cv2.boundingRect(biggestContour)

            print("contours:", len(contours))
            print("largest contour has ", len(biggestContour), "points")

            hull = cv2.convexHull(biggestContour)
            cv2.drawContours(imgContours, [hull], 0, (255, 255, 255), 1)
            print("convex hull has ", len(hull), "points")

            # epsilon = cv2.arcLength(biggestContour, True)
            # approx = cv2.approxPolyDP(biggestContour, 0.02 * epsilon, True)
            # cv2.drawContours(imgContours, [approx], 0, (255, 255, 255), 2)
            # print("simplified contour has", len(approx), "points")

        # Apply Canny edge detection
        imgContours = cv2.cvtColor(imgContours, cv2.COLOR_BGR2GRAY)
        # highThresh, thresh = cv2.threshold(imgContours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # lowThresh = 0.5 * highThresh
        imgCanny = cv2.Canny(imgContours, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(imgCanny, 2, np.pi / 360, 100)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 3000 * (-b)), int(y0 + 3000 * a))
                pt2 = (int(x0 - 3000 * (-b)), int(y0 - 3000 * a))
                cv2.line(imageCopy, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

        gtMaskCopy = cv2.cvtColor(gtMaskCopy, cv2.COLOR_BGR2RGB)
        predMask = cv2.cvtColor(predMask, cv2.COLOR_BGR2RGB)
        imgClosing = cv2.cvtColor(imgClosing, cv2.COLOR_BGR2RGB)
        imgCanny = cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2RGB)

        # Resize the images for visualization
        imageCopy = imutils.resize(imageCopy, height=700)
        gtMaskCopy = imutils.resize(gtMaskCopy, height=700)
        predMask = imutils.resize(predMask, height=700)
        imgClosing = imutils.resize(imgClosing, height=700)
        imgCanny = imutils.resize(imgCanny, height=700)


        # Cast the masks to float data type
        # gtMaskCopy = gtMaskCopy.astype("float32") / 255.0
        # predMask = predMask.astype("float32") / 255.0

        # Prepare a plot for visualization
        prepare_plot(imageCopy, gtMaskCopy, predMask, imgClosing, imgCanny)


# Load the image paths in our testing file and randomly select 10 image paths
print("[INFO] Loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
# imagePaths = imagePaths[140:149]
imagePaths = ['grc_passport_21.jpg', 'lva_passport_26.jpg', 'grc_passport_27.jpg', 'aze_passport_39.jpg',
              'lva_passport_21.jpg', 'lva_passport_17.jpg', 'alb_id_39.jpg', 'svk_id_36.jpg', 'lva_passport_35.jpg',
              'grc_passport_28.jpg', 'srb_passport_23.jpg', 'srb_passport_16.jpg', 'esp_id_47.jpg']
# imagePaths = np.random.choice(imagePaths, size=10)

# Load our model from disk and flash it to the current device
print("[INFO] Load up model...")
unet = torch.load('output/unet11_model_1/unet11_midv_2020_1e-3_10_16_256_personal.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet11_model_4/unet11_midv_2020_1e-3_30_8_256_colab_free.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet16_model_1/unet11_midv_2020_1e-3_30_8_256_kaggle.path', map_location=torch.device('cpu'))

# Iterate over the randomly selected test image paths
for path in imagePaths:
    # print(path)
    # Make predictions and visualize the results
    # make_predictions(unet, path)
    make_predictions(unet, 'dataset\\train\images\\' + path)
