import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# import the necessary packages
from lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import config
from squarepad import SquarePad

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('./output/model.pth')

# initialize our data augmentation functions
grayscale = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
	config.INPUT_WIDTH))

testTransforms = transforms.Compose([
	grayscale,
	SquarePad(), 
	resize, 
	transforms.ToTensor()
])

test_folder = 'test'

# initialize the training and validation dataset
print("[INFO] loading test dataset...")
testDataset = ImageFolder(root=test_folder, 
transform=testTransforms)

print("[INFO] test dataset contains {} samples...".format(len(testDataset)))

testDataLoader = DataLoader(testDataset, batch_size=1)


# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    preds = []

    # loop over the test set
    for (input, label) in testDataLoader:
        # send the input to the device
        (input, label) = (input.to(device), label.to(device))

        pred = model(input)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

        # if prediction failed
        if (pred.argmax(1) != label):
            newImg = torch.squeeze(input)
            imgplot = plt.imshow(newImg, cmap= 'gray')
            plt.show()

# generate a classification report
print(classification_report(np.array(testDataset.targets),
	np.array(preds), target_names=testDataset.classes))
        



