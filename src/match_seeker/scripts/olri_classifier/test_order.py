from paths import DATA
import numpy as np
from collections import OrderedDict


def getOneHotLabel(number,size):
    onehot = [0] * size
    onehot[number] = 1
    return onehot



if __name__ == '__main__':

    #This is testing that the labels are correctly labeled!
    cell_num = np.load(DATA + 'cell_ouput13k.npy')
    whichCell = 0
    for i in range(18, 44, 1):
        targetLabel = getOneHotLabel(i, 271)
        for j in range(500):
            if (targetLabel != cell_num[whichCell]).all():
                print("This is the target label", targetLabel)
                print("This is in the data", cell_num[whichCell])
                whichCell += 1

    print("The cells are correct!")

