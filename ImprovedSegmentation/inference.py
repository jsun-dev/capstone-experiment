import os
import cv2
import torch
import imutils
import random
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation():
    transforms = [
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms)

# Model settings
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['document']
ACTIVATION = 'sigmoid'

# Create the model
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# Load the parameters and weights
model.load_state_dict(torch.load('models/unet-resnet50.pth', map_location=torch.device('cpu')))

TEST_PATH = os.path.join('MIDV-2020', 'test')
TEST_IMAGES_PATH = os.path.join(TEST_PATH, 'images')
TEST_MASKS_PATH = os.path.join(TEST_PATH, 'masks')

TEST_IMAGES = os.listdir(TEST_IMAGES_PATH)
# random_sample = random.sample(TEST_IMAGES, 10)
for img_name in TEST_IMAGES:
    img_path = os.path.join(TEST_IMAGES_PATH, img_name)
    mask_path = os.path.join(TEST_MASKS_PATH, 'm_' + img_name.replace('jpg', 'png'))

    # Load the image and ground truth
    img = cv2.imread(img_path)
    gt_mask = cv2.imread(mask_path)

    # Get the original width and height
    height = img.shape[0]
    width = img.shape[1]

    # Convert the image as an input to the model
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)
    transform = get_augmentation()
    augmented = transform(image=img_input)
    img_input = augmented['image']
    img_input = torch.from_numpy(img_input.detach().cpu().numpy()).unsqueeze(0)

    # Get the predicted mask
    pr_mask = model.predict(img_input)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = cv2.threshold(pr_mask, 0, 255, cv2.THRESH_BINARY)[1]
    pr_mask = cv2.resize(pr_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # Show the image, ground truth, and predicted mask
    cv2.imshow('image', imutils.resize(img, height=700))
    cv2.imshow('truth', imutils.resize(gt_mask, height=700))
    cv2.imshow('prediction', imutils.resize(pr_mask, height=700))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Store the ground truth in a directory
    # print('Stored predicted mask of', img_name)
    # cv2.imwrite('output/m_' + img_name.replace('jpg', 'png'), pr_mask)
