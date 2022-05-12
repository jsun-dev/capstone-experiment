# Import the necessary package
import torch

# Specify the image dimension
IMAGE_SIZE = 224

# Specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Determine the device we will be using for inference
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify path to the MIDV-2020 labels
IN_LABELS = 'MIDV_2020/midv_2020_labels.txt'
