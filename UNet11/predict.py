# Import necessary packages
from itertools import combinations
from sklearn.cluster import KMeans
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import imutils
import math

# np.seterr(divide='ignore', invalid='ignore')

def get_intersections(lines, image):
    """Finds the intersections between groups of lines."""
    intersections = []
    group_lines = combinations(range(len(lines)), 2)
    x_in_range = lambda x: 0 <= x <= image.shape[1]
    y_in_range = lambda y: 0 <= y <= image.shape[0]

    for i, j in group_lines:
        line_i, line_j = lines[i][0], lines[j][0]

        if 30 < get_angle_between_lines(line_i, line_j) < 150:
            int_point = intersection(line_i, line_j)

            if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]):
                intersections.append(int_point)

    return intersections


def get_angle_between_lines(line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    angle = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        m1 = -(np.cos(theta1) / np.sin(theta1))
        m2 = -(np.cos(theta2) / np.sin(theta2))
        angle = abs(math.atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)
    return angle

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
      [np.cos(theta1), np.sin(theta1)],
      [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def find_quadrilaterals(intersections):
    X = np.array([[point[0][0], point[0][1]] for point in intersections])
    kmeans = KMeans(
        n_clusters=4,
        init='k-means++',
        max_iter=1000,
        n_init=10,
        random_state=0
    ).fit(X)

    return [[center.tolist()] for center in kmeans.cluster_centers_]

# https://stackoverflow.com/a/67686428
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

# Get width and height of aligned image
def get_width_and_height(pts):
    top_left = pts[0]
    top_right = pts[1]
    bottom_right = pts[2]
    bottom_left = pts[3]

    topWidth = math.hypot(top_left[1] - top_right[1], top_left[0] - top_right[0])
    bottomWidth = math.hypot(bottom_left[1] - bottom_right[1], bottom_left[0] - bottom_right[0])
    rightHeight = math.hypot(top_right[1] - bottom_right[1], top_right[0] - bottom_right[0])
    leftHeight = math.hypot(top_left[1] - bottom_left[1], top_left[0] - bottom_left[0])

    # width = (topWidth + bottomWidth) / 2
    # height = (rightHeight + leftHeight) / 2

    width = min(topWidth, bottomWidth)
    height = min(rightHeight, leftHeight)

    return int(width), int(height)

def prepare_plot(origImage, origMask, predMask, imgClosing, imgContours, imgResult):
    # Initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # Plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # ax[3].imshow(imgClosing)
    # ax[4].imshow(imgContours)

    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # ax[3].set_title("Morphology")
    # ax[4].set_title("Simplified Contour")

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
    cv2.imshow('result', imgResult)
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
        contours, hierarchy = cv2.findContours(imgClosing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process the contours
        if len(contours) != 0:
            # Assume the largest contour is the document
            sortedContours = sorted(contours, key=cv2.contourArea)
            biggestContour = sortedContours[-1]

            # Simplify the document contour with its convex hull
            hull = cv2.convexHull(biggestContour)
            cv2.drawContours(imgContours, [hull], 0, (255, 255, 255), 1)

            # epsilon = cv2.arcLength(biggestContour, True)
            # approx = cv2.approxPolyDP(biggestContour, 0.02 * epsilon, True)
            # cv2.drawContours(imgContours, [approx], 0, (255, 255, 255), 2)
            # print("simplified contour has", len(approx), "points")

        # Apply Canny edge detection
        imgCanny = cv2.Canny(imgContours, 50, 150, apertureSize=3)

        # Get Hough lines
        imgLines = imageCopy.copy()
        lines = cv2.HoughLines(imgCanny, 2, np.pi / 360, 100)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1500 * (-b)), int(y0 + 1500 * a))
                pt2 = (int(x0 - 1500 * (-b)), int(y0 - 1500 * a))
                cv2.line(imgLines, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

            # Get all intersections
            intersections = get_intersections(lines, imgLines)
            for inter in intersections:
                coord = inter[0]
                imgLines = cv2.circle(imgLines, (coord[0], coord[1]), radius=5, color=(0, 255, 0), thickness=-1)

            if len(intersections) < 4:
                imgResult = np.zeros((256, 256), dtype=np.uint8)
            else:
                # Get the most probable corners of the document using KMeans clustering
                quads = find_quadrilaterals(intersections)
                pts = []
                for q in quads:
                    qCoord = q[0]
                    pts.append((int(qCoord[0]), int(qCoord[1])))

                pts = np.array(pts, dtype='float32')

                # Order the corners
                orderedPts = order_points(pts)
                for p in orderedPts:
                    imgLines = cv2.circle(imgLines, (int(p[0]), int(p[1])), radius=10, color=(255, 255, 255),
                                          thickness=-1)

                # Process the corners to crop and align the document
                width, height = get_width_and_height(orderedPts)
                dst_pts = np.array([[0, 0],
                                    [width, 0],
                                    [width, height],
                                    [0, height]], dtype="float32")
                M = cv2.getPerspectiveTransform(orderedPts, dst_pts)
                imgResult = cv2.warpPerspective(imageCopy, M, (width, height))

        imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)
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
        imgLines = imutils.resize(imgLines, height=700)

        # Prepare a plot for visualization
        # prepare_plot(imageCopy, gtMaskCopy, predMask, imgClosing, imgCanny)
        prepare_plot(imgLines, gtMaskCopy, predMask, imgClosing, imgCanny, imgResult)


# Load the image paths in our testing file and randomly select 10 image paths
print("[INFO] Loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# imagePaths = imagePaths[140:149]

# Bad documents: 8/150 (Accuracy: 94.67%)
# Could potentially be improved to 5/150 bad documents (Accuracy: 96.67%) with data augmentation
# imagePaths = ['srb_passport_11.jpg', 'grc_passport_27.jpg', 'lva_passport_01.jpg', 'esp_id_89.jpg',
#               'grc_passport_22.jpg', 'grc_passport_28.jpg', 'esp_id_47.jpg', 'srb_passport_10.jpg']


# Load our model from disk and flash it to the current device
print("[INFO] Load up model...")
unet = torch.load('output/unet11_model_1/unet11_midv_2020_1e-3_10_16_256_personal.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet11_model_3/unet11_midv_2020_1e-3_20_32_256_colab_free.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet11_model_4/unet11_midv_2020_1e-3_30_8_256_colab_free.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet16_model_1/unet11_midv_2020_1e-3_30_8_256_kaggle.path', map_location=torch.device('cpu'))
# unet = torch.load('output/unet11_model_5/unet11_midv_2020_1e-3_15_4_256_kaggle.path', map_location=torch.device('cpu'))


# Iterate over the randomly selected test image paths
for path in imagePaths:
    # print(path)
    # Make predictions and visualize the results
    make_predictions(unet, path)
    # make_predictions(unet, 'dataset\\train\images\\' + path)
