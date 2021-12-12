import numpy as np
import torchvision.transforms.functional as F

# Apply padding to image to form a square image with the same height and width
# Needed to ensure that the proportions of the sculpture image is maintained
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')