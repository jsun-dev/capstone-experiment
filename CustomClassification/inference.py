# https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/
# Import necessary packages
import os
import random

import cv2
import torch
import config
import imutils
import numpy as np

def get_test_data(test_directory):
    data = []
    subdirs = os.listdir(test_directory)
    for sd in subdirs:
        sd_path = test_directory + '/' + sd
        files = os.listdir(sd_path)
        for f in files:
            data.append((sd_path + '/' + f, sd))

    return data

def preprocess_img(image):
    # Swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype('float32') / 255.0

    # Subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    image -= config.MEAN
    image /= config.STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)

    # Return the preprocessed image
    return image

# https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
def draw_text(img, text, pos=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX,
              scale=4, color=(0, 0, 255), thickness=6):
    # Get the text size
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size

    # Get the coordinates and dimension of the overlay
    x, y = pos
    pad = text_h // 2
    rect_w, rect_h = text_w, text_h

    # Draw the overlay
    sub_img = img[y:y+rect_h, x:x+rect_w]
    overlay = np.zeros(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.35, overlay, 0.35, 1.0)
    img[y:y+rect_h, x:x+rect_w] = res

    cv2.putText(img, text, (x, y + text_h + scale - 1), font, scale, color, thickness)

# Load the model and set it to evaluation mode
print('[INFO] Loading the model...')
# model = torch.load('midv_2020_model1/MIDV_2020_model.pt').to(config.DEVICE)
model = torch.load('midv_2020_model1/MIDV_2020_model.pt', map_location=torch.device('cpu'))
model.eval()

# Load the preprocessed the MIDV-2020 labels
print("[INFO] Loading MIDV-2020 labels...")
midv_2020_labels = dict(enumerate(open(config.IN_LABELS)))

# Get the test paths and labels
print("[INFO] Getting test data...")
test_data = get_test_data('MIDV_2020/test')
test_data = random.sample(test_data, 10)

# Start classifying the test images
print('[INFO] Classifying the images...')
for path, actual in test_data:
    # Load the image from disk, keep a copy, and preprocess it
    image = cv2.imread(path)
    orig = image.copy()
    image = preprocess_img(image)

    # Convert the preprocessed image to a torch tensor and move it
    # to the GPU if available
    image = torch.from_numpy(image)
    image = image.to(config.DEVICE)

    # Classify the image and extract the predictions
    logits = model(image)
    probabilities = torch.nn.Softmax(dim=-1)(logits)
    sortedProbabilities = torch.argsort(probabilities, dim=-1, descending=True)

    # Loop over the predictions and display the top-5 predictions and
    # corresponding probabilities to the terminal
    for (i, idx) in enumerate(sortedProbabilities[0, :5]):
        print("{}. {}: {:.2f}%".format(i + 1, midv_2020_labels[idx.item()].strip(),
                                       probabilities[0, idx.item()] * 100))
    print()

    # Draw the top prediction on the image and display the image
    (label, prob) = (midv_2020_labels[probabilities.argmax().item()],
                     probabilities.max().item())
    draw_text(orig, "Actual: {}".format(actual),
              pos=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, scale=3, color=(0, 255, 0), thickness=6)
    draw_text(orig, "Predict: {}, {:.2f}%".format(label.strip(), prob * 100),
              pos=(50, 160), font=cv2.FONT_HERSHEY_SIMPLEX, scale=3, color=(0, 0, 255), thickness=6)
    cv2.namedWindow(actual)
    cv2.moveWindow(actual, 500, 20)
    cv2.imshow(actual, imutils.resize(orig, height=700))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
