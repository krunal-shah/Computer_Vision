import numpy as np
import cv2

def calculatek(gray_image, means, variances):
	probab = (1/(np.square(variances) + 0.00001))*np.exp(-1*((np.square(gray_image-means))/2*(np.square(variances)+0.00001)))
	# ks = probab[1] > 0.00005
	# ks = ks*1
	ks = np.argmax(probab, 0)
	return ks
