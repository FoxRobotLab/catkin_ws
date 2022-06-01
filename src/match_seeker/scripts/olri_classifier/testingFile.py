import numpy as np
from paths import DATA

if __name__ == '__main__':
    mat = np.load(DATA + 'testNewMatrix.npy')

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 1:
                print("cells ", i, " and ", j, " are neighbors")
