# USAGE
# python3 train_dataset.py --model output/model.pth --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
# matplotlib.use("Agg")

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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize our data augmentation functions
grayscale = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
	config.INPUT_WIDTH))

# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([
	grayscale,
	SquarePad(), 
	resize, 
	transforms.ToTensor()
])
valTransforms = transforms.Compose([
	grayscale,
	SquarePad(), 
	resize, 
	transforms.ToTensor()
])
testTransforms = transforms.Compose([
	grayscale,
	SquarePad(), 
	resize, 
	transforms.ToTensor()
])

# initialize the training and validation dataset
print("[INFO] loading the training, validation and test dataset...")
trainDataset = ImageFolder(root=config.TRAIN,
	transform=trainTransforms)
valDataset = ImageFolder(root=config.VAL, 
	transform=valTransforms)
testDataset = ImageFolder(root=config.TEST, 
	transform=testTransforms)
print("[INFO] training dataset contains {} samples...".format(
	len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(
	len(valDataset)))
print("[INFO] test dataset contains {} samples...".format(
	len(testDataset)))

# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, batch_size=config.BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)
testDataLoader = DataLoader(testDataset, batch_size=config.BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // config.BATCH_SIZE
valSteps = len(valDataLoader.dataset) // config.BATCH_SIZE

# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(numChannels=1, classes=2).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=config.INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# visualize dataset
# matplotlib.use('tkagg')
# train_features, train_labels = next(iter(trainDataLoader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# img = img[0]
# label = train_labels[0]
# plt.imshow(img)
# plt.show()
# print(f"Label: {label}")

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, config.EPOCHS):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (input, label) in trainDataLoader:
		# send the input to the device
		(input, label) = (input.to(device), label.to(device))
		
		# perform a forward pass and calculate the training loss
		pred = model(input)
		loss = lossFn(pred, label)
		
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == label).type(
			torch.float).sum().item()

	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (input, label) in valDataLoader:
			# send the input to the device
			(input, label) = (input.to(device), label.to(device))

			# make the predictions and calculate the validation loss
			pred = model(input)
			totalValLoss += lossFn(pred, label)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == label).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (input, label) in testDataLoader:
		# send the input to the device
		input = input.to(device)

		# make the predictions and add them to the list
		pred = model(input)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
		
# generate a classification report
print(classification_report(np.array(testDataset.targets),
	np.array(preds), target_names=testDataset.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])