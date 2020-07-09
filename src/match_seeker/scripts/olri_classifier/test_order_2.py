from paths import DATA
import numpy as np
import cv2
from collections import OrderedDict




data = np.load(DATA + 'regressionTestSet.npz')

xyhOut = data['xyhOut']
images = data['images']


np.save(DATA + "regressionImages", images)
np.save(DATA + "regressionOutput", xyhOut)

# images = np.load(DATA + 'regressionImages.npy')
# print(np.shape(data['images']))
