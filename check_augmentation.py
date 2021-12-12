# set the device we will be using to train the model
from PIL import Image
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import config
from squarepad import SquarePad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = Image.open(r'./dataset/burmese/1E7686FD-DF75-4029-8509-2DBB3BF64442_4_5005_c.jpeg')

# initialize our data augmentation functions
grayscale = transforms.Grayscale(num_output_channels=1)
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
	config.INPUT_WIDTH))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
affine = transforms.RandomAffine(degrees=(-15,15))

# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([
    grayscale,
    SquarePad(), 
    resize,
    hFlip, 
    affine,
    transforms.ToTensor()
])

for i in range(5):
    newImg = trainTransforms(img)
    # remove dimension of size 1
    newImg = torch.squeeze(newImg)
    imgplot = plt.imshow(newImg, cmap= 'gray')
    plt.show()