import os
import random

TEST_SPLIT = 0.2

# Get the subdirectories from the train directory
subdirs = os.listdir('train')

# Create a test directory and populate each subdirectory with split files
os.makedirs('test')
for sd in subdirs:
    # Create the subdirectory
    os.makedirs('test/' + sd)

    # Get all files from the current training subdirectory
    files = os.listdir('train/' + sd)

    # Split the files by with 20% being for testing
    test_files = random.sample(files, int(len(files) * TEST_SPLIT))

    # Move the testing files from training subdirectory to testing subdirectory
    for tf in test_files:
        os.rename('train/' + sd + '/' + tf, 'test/' + sd + '/' + tf)