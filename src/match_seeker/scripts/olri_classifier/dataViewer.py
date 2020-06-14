

import cv2
import numpy as np


data = np.load("../../res/classifier2019data/NEWTRAININGDATA_100_500withCellInput95k.npy", allow_pickle=True, encoding='latin1')

print(data.shape, data.dtype)


images = data[:, 1]
classes = data[:, 1]

print(images[0].shape, type(images), type(images[0]), images.shape, images.dtype)
print(images[1].shape)
#print(classes[0])

textfile = open('textfile.txt', 'w')

for i in range(20):
	im  = images[i]
	img = im[:,:,1,]
	textfile.write(str(img))
textfile.close()
