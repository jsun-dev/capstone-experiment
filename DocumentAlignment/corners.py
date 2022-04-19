# Reference: https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65
import cv2
import imutils
import numpy as np
import math

from itertools import combinations
from sklearn.cluster import KMeans

# # Callback function for trackbars
# def nothing(x):
#     pass
#
# # Initialize trackbars
# def initializeTrackbars():
#     cv2.namedWindow("Trackbars")
#     cv2.resizeWindow("Trackbars", 360, 240)
#     cv2.createTrackbar("Threshold1", "Trackbars", 110, 255, nothing)
#     cv2.createTrackbar("Threshold2", "Trackbars", 0, 255, nothing)
#
# # Get trackbar values
# def valueTrackbars():
#     thresh1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
#     thresh2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
#     return thresh1, thresh2

# initializeTrackbars()

def get_intersections(lines, image):
    """Finds the intersections between groups of lines."""
    intersections = []
    group_lines = combinations(range(len(lines)), 2)
    x_in_range = lambda x: 0 <= x <= image.shape[1]
    y_in_range = lambda y: 0 <= y <= image.shape[0]

    for i, j in group_lines:
        line_i, line_j = lines[i][0], lines[j][0]

        if 45 < get_angle_between_lines(line_i, line_j) < 135.0:
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
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(math.atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)

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
        n_clusters = 8,
        init = 'k-means++',
        max_iter = 100,
        n_init = 10,
        random_state = 0
    ).fit(X)

    return [[center.tolist()] for center in kmeans.cluster_centers_]

# https://stackoverflow.com/a/67686428
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
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

    return (int(width), int(height))

# Read the image
imgOriginal = cv2.imread('warp_test.jpg')
imgOriginal = imutils.resize(imgOriginal, width=800)

# Convert original image to grayscale and calculate the brightness
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
cols, rows = imgGray.shape
brightness = np.sum(imgGray) / (255 * cols * rows)
print('Brightness:', brightness)

# thresh = valueTrackbars()
# Linear equation of level threshold based on brightness
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

        intersections = get_intersections(lines, imgOriginal)
        for inter in intersections:
            coord = inter[0]
            imgLines = cv2.circle(imgLines, (coord[0], coord[1]), radius=5, color=(0, 255, 0), thickness=-1)

        quads = find_quadrilaterals(intersections)
        pts = []
        for q in quads:
            qCoord = q[0]
            pts.append((int(qCoord[0]), int(qCoord[1])))

        pts = np.array(pts, dtype='float32')

        orderedPts = order_points(pts)
        for p in orderedPts:
            imgLines = cv2.circle(imgLines, (int(p[0]), int(p[1])), radius=10, color=(255, 255, 255),thickness=-1)

        width, height = get_width_and_height(orderedPts)
        dst_pts = np.array([[0, 0],
                            [width, 0],
                            [width, height],
                            [0, height]], dtype="float32")
        M = cv2.getPerspectiveTransform(orderedPts, dst_pts)
        imgResult = cv2.warpPerspective(imgOriginal, M, (width, height))

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
        cv2.imshow('result', imgResult)
        cv2.waitKey(0)
    else:
        print('Cannot detect document')
else:
    print('Cannot detect document')
