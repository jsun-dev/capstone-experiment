# Reference: https://www.geeksforgeeks.org/image-registration-using-opencv-python/
#            https://pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

import cv2
import imutils
import numpy as np

# Import images
img1_color = cv2.imread('test7.jpg')
img2_color = cv2.imread('ref.jpg')
img1_color = imutils.resize(img1_color, width=800)

# Convert to grayscale
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

# Get dimension of reference image
height, width = img2.shape

# Create ORB detector with 5000 features
orb = cv2.ORB_create(500)

# Find keypoints and descriptors
kp1, d1 = orb.detectAndCompute(img1, None)
kp2, d2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
matcher = cv2.DescriptorMatcher_create(method)

# Match descriptors
matches = matcher.match(d1, d2, None)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

keep = int(len(matches) * 0.2)
matches = matches[:keep]

matchedVis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
cv2.imshow("Matched Keypoints", matchedVis)
cv2.waitKey(0)

# Define empty matrices of shape numMatches * 2
p1 = np.zeros((len(matches), 2), dtype=np.float32)
p2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, m in enumerate(matches):
    p1[i] = kp1[m.queryIdx].pt
    p2[i] = kp2[m.trainIdx].pt

# Find the homography matrix
homography, mask = cv2.findHomography(p1, p2, method=cv2.RANSAC)

# Use this matrix to transform the colored image wrt the reference image
aligned = cv2.warpPerspective(img1_color, homography, (width, height))

# Show the result
cv2.imshow('result', aligned)
cv2.waitKey(0)
