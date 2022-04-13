# Based on: https://www.youtube.com/watch?v=ON_JubFRw8M

import cv2
import imutils
import numpy as np

# Callback function for trackbars
def nothing(x):
    pass

# Initialize trackbars
def initializeTrackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)

# Get trackbar values
def valueTrackbars():
    thresh1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return thresh1, thresh2

# Get biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# Reorder points
def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

#initializeTrackbars()

while True:
    # Import images
    img1 = cv2.imread('test1.jpg')
    img2 = cv2.imread('ref.jpg')

    # Basic pre-processing
    img = imutils.resize(img1, width=800)               # Resize the image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)      # Add blur

    # Apply thresholding
    ret, imgThresh = cv2.threshold(imgBlur, 200, 255, cv2.THRESH_BINARY_INV)
    #imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)

    # Apply morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    imgClosing = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)
    # imgDilation = cv2.dilate(imgThresh, kernel, iterations=2)
    # imgErosion = cv2.erode(imgDilation, kernel, iterations=1)

    # Apply Canny edge detection
    # thresh = valueTrackbars()
    # imgCanny = cv2.Canny(imgErosion, thresh[0], thresh[1])

    # Find all contours
    imgContours = img.copy()
    imgDocumentContour = img.copy()
    contours, hierarchy = cv2.findContours(imgClosing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # Draw all contours in blue
        cv2.drawContours(imgContours, contours, -1, 255, 3)

        # Assume the second-largest contour is the document
        sortedContours = sorted(contours, key=cv2.contourArea)
        c = sortedContours[-2]
        x, y, w, h = cv2.boundingRect(c)

        # Draw the document contour in green
        cv2.rectangle(imgDocumentContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the fitted document contour in red
        rect = cv2.minAreaRect(c)
        (x, y), (rWidth, rHeight), angle = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imgDocumentContour, [box], 0, (0, 0, 255), 2)

        # Straighten and crop according to the document contour
        # Reference: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        imgResult = cv2.warpPerspective(img, M, (width, height))

        # reordered = reorder(box)
        # pts1 = np.float32(reordered)
        # pts2 = None
        # dimension = None
        # if rWidth > rHeight:
        #     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        #     dimension = (width, height)
        # else:
        #     pts2 = np.float32([[0, 0], [height, 0], [0, width], [height, width]])
        #     dimension = (height, width)
        #
        # matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # imgResult = cv2.warpPerspective(img, matrix, dimension)

    cv2.imshow('original', img)
    cv2.imshow('thresh', imgThresh)
    cv2.imshow('closing', imgClosing)
    # cv2.imshow('canny', imgCanny)
    # cv2.imshow('erosion', imgErosion)
    # cv2.imshow('dilation', imgDilation)

    cv2.imshow('contours', imgContours)
    cv2.imshow('documentContour', imgDocumentContour)
    cv2.imshow('result', imgResult)
    cv2.waitKey(1)
