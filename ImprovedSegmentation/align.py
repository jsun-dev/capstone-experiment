from itertools import combinations
from sklearn.cluster import KMeans
import sys
import os
import cv2
import math
import random
import imutils
import numpy as np
import scipy.spatial.distance
import time

def ed2(lhs, rhs):
    return(lhs[0] - rhs[0])*(lhs[0] - rhs[0]) + (lhs[1] - rhs[1])*(lhs[1] - rhs[1])

def remove_from_contour(contour, defectsIdx, tmp):
    minDist = sys.maxsize
    startIdx, endIdx = 0, 0

    for i in range(0,len(defectsIdx)):
        for j in range(i+1, len(defectsIdx)):
            dist = ed2(contour[defectsIdx[i]][0], contour[defectsIdx[j]][0])
            if minDist > dist:
                minDist = dist
                startIdx = defectsIdx[i]
                endIdx = defectsIdx[j]

    if startIdx <= endIdx:
        inside = contour[startIdx:endIdx]
        len1 = 0 if inside.size == 0 else cv2.arcLength(inside, False)
        outside1 = contour[0:startIdx]
        outside2 = contour[endIdx:len(contour)]
        len2 = (0 if outside1.size == 0 else cv2.arcLength(outside1, False)) + (0 if outside2.size == 0 else cv2.arcLength(outside2, False))
        if len2 < len1:
            startIdx,endIdx = endIdx,startIdx
    else:
        inside = contour[endIdx:startIdx]
        len1 = 0 if inside.size == 0 else cv2.arcLength(inside, False)
        outside1 = contour[0:endIdx]
        outside2 = contour[startIdx:len(contour)]
        len2 = (0 if outside1.size == 0 else cv2.arcLength(outside1, False)) + (0 if outside2.size == 0 else cv2.arcLength(outside2, False))
        if len1 < len2:
            startIdx,endIdx = endIdx,startIdx

    if startIdx <= endIdx:
        out = np.concatenate((contour[0:startIdx], contour[endIdx:len(contour)]), axis=0)
    else:
        out = contour[endIdx:startIdx]
    return out

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

    width = (topWidth + bottomWidth) / 2
    height = (rightHeight + leftHeight) / 2
    #
    # width = max(topWidth, bottomWidth)
    # height = max(rightHeight, leftHeight)

    return int(width), int(height)

def get_cardinal(pts, direction):
    sums = []
    for p in pts:
        if direction == 'x':
            sums.append(p[0])
        elif direction == 'x+y':
            sums.append(p[0] + p[1])
        elif direction == 'y':
            sums.append(p[1])
        elif direction == '-x+y':
            sums.append(-p[0] + p[1])
        elif direction == '-x':
            sums.append(-p[0])
        elif direction == '-x-y':
            sums.append(-p[0] - p[1])
        elif direction == '-y':
            sums.append(-p[1])
        elif direction == 'x-y':
            sums.append(p[0] - p[1])
    max_val = max(sums)
    max_idx = sums.index(max_val)
    return pts[max_idx]

TEST_PATH = os.path.join('MIDV-2020', 'test')
TEST_IMAGES_PATH = os.path.join(TEST_PATH, 'images')
TEST_IMAGES = os.listdir(TEST_IMAGES_PATH)

# masks = os.listdir('output')
# masks = ['m_est_id_03.png']

for name in TEST_IMAGES:
    print(name)
    img = cv2.imread(os.path.join(TEST_IMAGES_PATH, name))
    img = cv2.copyMakeBorder(img, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_copy = img.copy()
    mask = cv2.imread('output/m_' + name.replace('jpg', 'png'))
    mask = cv2.copyMakeBorder(mask, 300, 300, 300, 300, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contour = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    while True:
        defects_idx = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            depth = d / 256
            if depth > 40:
                defects_idx.append(f)

        if len(defects_idx) < 2:
            break

        cnt = remove_from_contour(cnt, defects_idx, mask)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

    hull = cv2.convexHull(cnt)
    cv2.drawContours(contour, [hull], 0, (255, 255, 255), 12)
    contour = cv2.cvtColor(contour, cv2.COLOR_BGR2GRAY)

    pts = []
    for h in hull:
        pts.append(tuple(h[0]))

    cardinals = []
    cardinals.append(get_cardinal(pts, 'x'))
    cardinals.append(get_cardinal(pts, 'x+y'))
    cardinals.append(get_cardinal(pts, 'y'))
    cardinals.append(get_cardinal(pts, '-x+y'))
    cardinals.append(get_cardinal(pts, '-x'))
    cardinals.append(get_cardinal(pts, '-x-y'))
    cardinals.append(get_cardinal(pts, '-y'))
    cardinals.append(get_cardinal(pts, 'x-y'))

    cardinal_cnt = np.array(cardinals)

    M = cv2.moments(cardinal_cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    hull_norm = cardinal_cnt - [cx, cy]
    hull_scaled = hull_norm * 1.5
    hull_scaled = hull_scaled + [cx, cy]
    hull_scaled = hull_scaled.astype(np.int32)

    cv2.drawContours(img, [hull_scaled], 0, (0, 255, 0), 12)

    lines = []
    for i in range(len(cardinals)):
        if i == len(cardinals) - 1:
            l = (cardinals[i], cardinals[0])
            lines.append((l, math.dist(l[0], l[1])))
        else:
            l = (cardinals[i], cardinals[i + 1])
            lines.append((l, math.dist(l[0], l[1])))

    # for l in lines:
    #     cv2.line(mask, l[0][0], l[0][1], (255, 0, 0), 12)

    sides = np.zeros(mask.shape, dtype=np.uint8)
    sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
    for i in range(4):
        sl = sorted_lines[i][0]
        cv2.line(sides, sl[0], sl[1], (255, 255, 255), 12)
    sides = cv2.cvtColor(sides, cv2.COLOR_BGR2GRAY)

    # Get Hough lines
    hough = cv2.HoughLines(sides, 1, np.pi / 180, 300, None, 0, 0)
    if hough is not None:
        for i in range(0, len(hough)):
            rho = hough[i][0][0]
            theta = hough[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * a))
            pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * a))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    intersections = get_intersections(hough, img)
    good_intersections = []
    for inter in intersections:
        coord = inter[0]
        pt = (coord[0], coord[1])
        within = cv2.pointPolygonTest(hull_scaled, pt, False)
        if within >= 0:
            good_intersections.append(inter)
            cv2.circle(img, pt, radius=20, color=(0, 255, 0), thickness=-1)
        else:
            cv2.circle(img, pt, radius=20, color=(255, 0, 0), thickness=-1)

    quads = find_quadrilaterals(good_intersections)
    pts = []
    for q in quads:
        qCoord = q[0]
        pts.append((int(qCoord[0]), int(qCoord[1])))

    pts = np.array(pts, dtype='float32')

    orderedPts = order_points(pts)
    for p in orderedPts:
        cv2.circle(img, (int(p[0]), int(p[1])), radius=40, color=(255, 255, 255), thickness=-1)

    # Process the corners to crop and align the document
    width, height = get_width_and_height(orderedPts)
    dst_pts = np.array([[0, 0],
                        [width, 0],
                        [width, height],
                        [0, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(orderedPts, dst_pts)
    result = cv2.warpPerspective(img_copy, M, (width, height))

    cv2.imshow('contour', imutils.resize(contour, height=700))
    cv2.imshow('sides', imutils.resize(sides, height=700))
    cv2.imshow('mask', imutils.resize(mask, height=700))
    cv2.imshow('image', imutils.resize(img, height=700))
    cv2.imshow('result', imutils.resize(result, width=700))
    cv2.waitKey(0)
    cv2.destroyAllWindows()