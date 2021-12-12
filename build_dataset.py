# Responsible fpr dividing and structuring the dataset into a training and validation set
from imutils import paths
import numpy as np
import shutil
import os
import config

def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)

	# loop over the image paths
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)

		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)

# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.SCULPTURE_DATASET_PATH))
np.random.shuffle(imagePaths)

# generate training, test and validation paths
testPathsLen = int(len(imagePaths) * config.TEST_SPLIT)
remainingPathsLen = len(imagePaths) - testPathsLen 
valPathsLen = int(remainingPathsLen * config.VAL_SPLIT)
trainPathsLen = remainingPathsLen - valPathsLen

testPaths = imagePaths[:testPathsLen]
valPaths = imagePaths[testPathsLen:testPathsLen + valPathsLen]
trainPaths = imagePaths[testPathsLen + valPathsLen:]

# copy the training and validation images to their respective
# directories
print("[INFO] copying training and validation images...")
print(f'[INFO] Training images:{trainPathsLen}\nValidation images: {valPathsLen}\nTest images: {testPathsLen}')
copy_images(trainPaths, config.TRAIN)
copy_images(valPaths, config.VAL)
copy_images(testPaths, config.TEST)