import os

# Get the subdirectories of the BID dataset
subdirs = os.listdir('BID Dataset')
subdirs.remove('desktop.ini')

# Extract the images to the BID directory
os.makedirs('BID')

# Start extracting images
for sd in subdirs:
    # Create the subdirectory
    os.makedirs('BID/' + sd)

    # Get all files from the current subdirectory
    files = os.listdir('BID Dataset/' + sd)

    # Move the image to the BID directory
    for f in files:
        if 'in.jpg' in f:
            os.rename('BID Dataset/' + sd + '/' + f, 'BID/' + sd + '/' + f)
