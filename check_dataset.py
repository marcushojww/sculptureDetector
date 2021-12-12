# Responsible fpr dividing and structuring the dataset into a training and validation set
from imutils import paths
import config

# check images in dataset folder
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.MASK_DATASET_PATH))
withoutMaskImagePaths = list(paths.list_images(config.WITHOUT_MASK_PATH))
incorrectMaskImagePaths = list(paths.list_images(config.INCORRECT_MASK_PATH))
correctMaskImagePaths = list(paths.list_images(config.CORRECT_MASK_PATH))


trainImagePaths = list(paths.list_images('train'))
valImagePaths = list(paths.list_images('val'))
testImagePaths = list(paths.list_images('test'))

# check images in test folder
# testCorrectMaskImagePaths = list(paths.list_images(f'test/with_mask'))
# testIncorrectMaskImagePaths = list(paths.list_images(f'test/incorrect_mask'))
# testWithoutMaskImagePaths = list(paths.list_images(f'test/without_mask'))


# print(f'[INFO] Number of images in dataset/without_mask folder: {len(withoutMaskImagePaths)}')
# print(f'[INFO] Number of images in dataset/incorrect_mask folder: {len(incorrectMaskImagePaths)}')
# print(f'[INFO] Number of images in dataset/correct_mask folder: {len(correctMaskImagePaths)}')

print(f'[INFO] Number of images in train folder: {len(trainImagePaths)}')
print(f'[INFO] Number of images in val folder: {len(valImagePaths)}')
print(f'[INFO] Number of images in test folder: {len(testImagePaths)}')

# print(f'[INFO] Number of images in test/without_mask folder: {len(testWithoutMaskImagePaths)}')
# print(f'[INFO] Number of images in test/incorrect_mask folder: {len(testIncorrectMaskImagePaths)}')
# print(f'[INFO] Number of images in test/correct_mask folder: {len(testCorrectMaskImagePaths)}')