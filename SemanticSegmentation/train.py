# USAGE
# python train.py

# Import necessary packages
from src.dataset import SegmentationDataset
from src.model import UNet
from src import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

if __name__ == '__main__':
    # Load the image and mask filepaths in a sorted manner
    imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

    # Partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)

    # Unpack the data split
    trainImages, testImages = split[:2]
    trainMasks, testMasks = split[2:]

    # Write the testing image paths to disk so that we can use them
    # when evaluating/testing our model
    print("[INFO] Saving testing image paths...")
    f = open(config.TEST_PATHS, 'w')
    f.write('\n'.join(testImages))
    f.close()

    # Define transformations
    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                        config.INPUT_IMAGE_WIDTH)),
                                     transforms.ToTensor()])

    # Create the train and test datasets
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)
    testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks, transforms=transforms)
    print(f"[INFO] Found {len(trainDS)} examples in the training set...")
    print(f"[INFO] Found {len(testDS)} examples in the test set...")

    # Create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                             num_workers=os.cpu_count())
    testLoader = DataLoader(testDS, shuffle=False,
                            batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
                            num_workers=os.cpu_count())

    # Initialize our U-Net model
    unet = UNet().to(config.DEVICE)

    # Initialize the loss function and optimizer
    lossFunc = BCEWithLogitsLoss()
    opt = Adam(unet.parameters(), lr=config.INIT_LR)

    # Calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    # Initialize a dictionary to store training history
    H = {'train_loss': [], 'test_loss': []}

    # Loop over epochs
    print("[INFO] Training the network...")
    startTime = time.time()
    for e in tqdm(range(config.NUM_EPOCHS)):
        # Set the model in training mode
        unet.train()

        # Initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

        # Loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # Send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # Perform a forward pass and calculate the training loss
            pred = unet(x)
            loss = lossFunc(pred, y)

            # First, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Add the loss to the total training loss so far
            totalTrainLoss += loss

        # Switch off autograd
        with torch.no_grad():
            # Set the model in evaluation mode
            unet.eval()

            # Loop over the validation set
            for (x, y) in testLoader:
                # Send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # Make the predictions and calculate the validation loss
                pred = unet(x)
                totalTestLoss += lossFunc(pred, y)

            # Calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps

            # Update our training history
            H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
            H['test_loss'].append(avgTestLoss.cpu().detach().numpy())

            # Print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
            print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

    # Display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime - startTime))

    # Plot the training loss
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(H['train_loss'], label='train_loss')
    plt.plot(H['test_loss'], label='test_loss')
    plt.title('Training Loss on Dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.savefig(config.PLOT_PATH)

    # Serialize the model to disk
    torch.save(unet, config.MODEL_PATH)
