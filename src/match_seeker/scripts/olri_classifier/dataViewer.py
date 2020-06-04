

import cv2
import numpy as np


data = np.load("../../res/classifier2019data/NEWTRAININGDATA_100_500withCellInput95k.npy")

#print(data.shape, data.dtype)


images = data[:, 0]
classes = data[:, 1]

#print(images[0])
#print(classes[0])

textfile = open('textfile.txt', 'w')

for i in range(20):
	im  = images[i]
	img = im[:,:,1,]
	textfile.write(str(img))
textfile.close()
