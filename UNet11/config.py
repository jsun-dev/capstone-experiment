# Import necessary packages
import torch
import os

# Base path of the dataset
DATASET_PATH = os.path.join('dataset', 'train')

# Define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, 'images')
MASK_DATASET_PATH = os.path.join(DATASET_PATH, 'masks')

# Define the test split
TEST_SPLIT = 0.15

# Determine the device to be used for training and evaluation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == 'cuda' else False

# Define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# HYPERPARAMETERS
# Initialize learning rate, number of epochs, and batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 16

# Define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# Define threshold to filter weak predictions
THRESHOLD = 0.5

# Define the path to the base output directory
BASE_OUTPUT = 'output'

# Define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, 'unet11_midv_2020.path')
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])
