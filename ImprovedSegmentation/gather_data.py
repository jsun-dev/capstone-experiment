# Import necessary packages
import os
import json
import cv2
import imutils
import numpy as np

# Base path of the MIDV-2020 dataset
MIDV_2020_PATH = os.path.join('MIDV-2020', 'photo')

# Define the path to the annotations and images
ANNOTATIONS_PATH = os.path.join(MIDV_2020_PATH, 'annotations')
IMAGES_PATH = os.path.join(MIDV_2020_PATH, 'images')

# Get all JSON file names
JSON_FILES = os.listdir(ANNOTATIONS_PATH)

# Get all subdirectory names in the images directory
IMAGE_LABELS = os.listdir(IMAGES_PATH)

# Get all mask coordinates
print("[INFO] Reading coordinates...")
masks = []
for fileName in JSON_FILES:
    # Open the file
    f = open(os.path.join(ANNOTATIONS_PATH, fileName))

    # Load the JSON data
    data = json.load(f)
    imgMetadata = data['_via_img_metadata']

    # Close the file
    f.close()

    # Group the coordinates by each image category
    maskCategories = []
    for i in imgMetadata:
        polygon = imgMetadata[i]['regions'][1]['shape_attributes']
        xCoords = polygon['all_points_x']
        yCoords = polygon['all_points_y']
        coords = list(zip(xCoords, yCoords))
        maskCategories.append(coords)

    # Collect every masks under an array
    masks.append(maskCategories)

# Create the images and masks directory
os.makedirs('images')
os.makedirs('masks')

# Create the mask for each image
for i in range(len(IMAGE_LABELS)):
    # Get image files
    label = IMAGE_LABELS[i]
    directory = os.path.join(IMAGES_PATH, label)
    files = os.listdir(directory)

    print("[INFO] Creating masks for ", label, "...", sep='')

    # Create the mask
    for j in range(len(files)):
        # Read the image
        file = files[j]
        file_path = os.path.join(directory, file)
        img = cv2.imread(file_path)

        # Create the mask from the coordinates
        mask = np.zeros(img.shape, dtype=np.uint8)
        coords = np.array(masks[i][j], dtype=np.int32)
        cv2.fillPoly(mask, coords.reshape(-1, 4, 2), color=(255, 255, 255))

        # Create the file names
        imgFileName = label + '_' + file
        maskFileName = 'm_' + imgFileName

        # Change mask from JPEG to PNG
        maskFileName = maskFileName.replace('jpg', 'png')

        # Write the image and mask to their respective subdirectory
        os.rename(file_path, os.path.join('images', imgFileName))
        cv2.imwrite(os.path.join('masks', maskFileName), cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

print("[INFO] Done...")
