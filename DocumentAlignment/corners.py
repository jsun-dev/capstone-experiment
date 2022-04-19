# Reference: https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65
import cv2
import imutils
import numpy as np
import math

# Callback function for trackbars
def nothing(x):
    pass

# Initialize trackbars
def initializeTrackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 110, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 0, 255, nothing)

# Get trackbar values
def valueTrackbars():
    thresh1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return thresh1, thresh2

# initializeTrackbars()

# Read the image
imgOriginal = cv2.imread('warp_test.jpg')
imgOriginal = imutils.resize(imgOriginal, width=800)

# Convert original image to grayscale and calculate the brightness
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
cols, rows = imgGray.shape
brightness = np.sum(imgGray) / (255 * cols * rows)
print('Brightness:', brightness)

# thresh = valueTrackbars()
# Linear equation of adjusted threshold based on brightness
levelThreshold = 300 * brightness + 10
print('Level Threshold:', levelThreshold)

# Adjust the levels of the image
# https://stackoverflow.com/a/60339950
inBlack = np.array([levelThreshold, levelThreshold, levelThreshold], dtype=np.float32)
inWhite = np.array([255, 255, 255], dtype=np.float32)
inGamma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
outBlack = np.array([0, 0, 0], dtype=np.float32)
outWhite = np.array([255, 255, 255], dtype=np.float32)

imgAdjusted = np.clip((imgOriginal - inBlack) / (inWhite - inBlack), 0, 255)
imgAdjusted = (imgAdjusted ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
imgAdjusted = np.clip(imgAdjusted, 0, 255).astype(np.uint8)

# Convert adjusted image to grayscale
imgAdjustedGray = cv2.cvtColor(imgAdjusted, cv2.COLOR_BGR2GRAY)

# Denoise the image
imgDenoise = cv2.fastNlMeansDenoising(imgAdjustedGray, h=7)

# Apply thresholding
ret, imgThresh = cv2.threshold(imgDenoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to erase unnecessary details
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
imgOpening = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel, iterations=1)
imgClosing = cv2.morphologyEx(imgOpening, cv2.MORPH_CLOSE, kernel, iterations=10)

# Apply Canny edge detection
imgCanny = cv2.Canny(imgClosing, 50, 150, apertureSize=3)

# Find all contours
imgContours = imgOriginal.copy()
imgDocumentContour = imgOriginal.copy()
contours, hierarchy = cv2.findContours(imgClosing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) != 0:
    # Draw all contours in blue
    cv2.drawContours(imgContours, contours, -1, 255, 3)

    # Assume the largest contour is the document
    sortedContours = sorted(contours, key=cv2.contourArea)
    biggestContour = sortedContours[-1]
    x, y, w, h = cv2.boundingRect(biggestContour)

    # Draw the document contour in green
    cv2.rectangle(imgDocumentContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Get Hough lines
    imgLines = imgOriginal.copy()#[y:y + h, x:x + w]
    imgCannyCropped = imgCanny.copy()[y:y + h, x:x + w]
    lines = cv2.HoughLines(imgCanny, 2, np.pi / 360, 100)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(imgLines, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('original', imgOriginal)
cv2.imshow('gray', imgGray)
cv2.imshow('adjusted', imgAdjusted)
cv2.imshow('adjustedGray', imgAdjustedGray)
# cv2.imshow('denoise', imgDenoise)
cv2.imshow('threshold', imgThresh)
cv2.imshow('closing', imgClosing)
cv2.imshow('canny', imgCanny)
cv2.imshow('contours', imgContours)
cv2.imshow('document', imgDocumentContour)
cv2.imshow('lines', imgLines)
cv2.waitKey(0)
