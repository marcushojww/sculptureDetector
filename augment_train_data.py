from PIL import Image
from imutils import paths
from torchvision import transforms
import torch
import config
import os
from squarepad import SquarePad

img = Image.open(r'./dataset/burmese/1E7686FD-DF75-4029-8509-2DBB3BF64442_4_5005_c.jpeg')

imagePaths = list(paths.list_images(config.TRAIN))

hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.15)
affine = transforms.RandomAffine(degrees=(-20,20), translate=(0.1,0.1))

augmentation_transform = transforms.Compose([
    hFlip, 
    vFlip,
    affine
])

for path in imagePaths:
    # add five augmented images
    img = Image.open(path, mode='r')
    imgName = path.split(os.path.sep)[-1]
    label = path.split(os.path.sep)[-2]
    for i in range(5):
        aug_image = augmentation_transform(img)
        aug_image.save(f'./train/{label}/{imgName}_aug{i}.jpeg')
