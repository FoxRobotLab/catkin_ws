import numpy as np
from paths import DATA

if __name__ == '__main__':
    mat = np.load(DATA + 'testNewMatrix.npy')
    print(mat)
