from paths import DATA
import numpy as np
import cv2
from collections import OrderedDict




data = np.load(DATA + 'regressionTestSet.npz')

xyhOut = np.load(data['xyhOut'])

print(xyhOut)

