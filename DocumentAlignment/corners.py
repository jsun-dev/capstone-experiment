# Reference: https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65
import cv2
import imutils
import numpy as np
import math

imgOriginal = cv2.imread('warp_test.jpg')
imgOriginal = imutils.resize(imgOriginal, width=800)

# Convert to grayscale
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

# Denoise the image
imgDenoise = cv2.fastNlMeansDenoising(imgGray, h=7)

# Apply thresholding
ret, imgThresh = cv2.threshold(imgDenoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#ret, imgThresh = cv2.threshold(imgDenoise, 200, 255, cv2.THRESH_BINARY)

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
    imgLines = imgOriginal.copy()[y:y + h, x:x + w]
    imgCannyCropped = imgCanny.copy()[y:y + h, x:x + w]
    lines = cv2.HoughLines(imgCannyCropped, 2, np.pi / 360, 100)
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
    # cv2.imshow('gray', imgGray)
    # cv2.imshow('denoise', imgDenoise)
    cv2.imshow('threshold', imgThresh)
    cv2.imshow('closing', imgClosing)
    cv2.imshow('canny', imgCanny)
    cv2.imshow('contours', imgContours)
    cv2.imshow('document', imgDocumentContour)
    cv2.imshow('lines', imgLines)
    cv2.waitKey(0)
