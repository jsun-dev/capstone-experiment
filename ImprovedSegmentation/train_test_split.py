import os
import random

TEST_SPLIT = 0.2

TRAIN_PATH = os.path.join('MIDV-2020', 'train')
TRAIN_IMAGES_PATH = os.path.join(TRAIN_PATH, 'images')
TRAIN_MASKS_PATH = os.path.join(TRAIN_PATH, 'masks')

TEST_PATH = os.path.join('MIDV-2020', 'test')
TEST_IMAGES_PATH = os.path.join(TEST_PATH, 'images')
TEST_MASKS_PATH = os.path.join(TEST_PATH, 'masks')

TOTAL_DATA = len(os.listdir(TRAIN_IMAGES_PATH))
TOTAL_INDICES = list(range(0, TOTAL_DATA))
RANDOM_INDICES = random.sample(TOTAL_INDICES, int(len(TOTAL_INDICES) * TEST_SPLIT))

TRAIN_IMAGES = os.listdir(TRAIN_IMAGES_PATH)
TRAIN_MASKS = os.listdir(TRAIN_MASKS_PATH)

for i in RANDOM_INDICES:
    image = TRAIN_IMAGES[i]
    mask = TRAIN_MASKS[i]
    os.rename(os.path.join(TRAIN_IMAGES_PATH, image), os.path.join(TEST_IMAGES_PATH, image))
    os.rename(os.path.join(TRAIN_MASKS_PATH, mask), os.path.join(TEST_MASKS_PATH, mask))