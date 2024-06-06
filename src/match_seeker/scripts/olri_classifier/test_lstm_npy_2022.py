from paths import DATA
import numpy as np
import matplotlib as plt


class loadNpy(object):

    def __init__(self):
        self.lstmDataFile = DATA + "Img_122k_ordered.npy"
        self.lstmCellOutput = DATA + "lstm_cellOutput_122k.npy"
        self.lstmHeadOutput = DATA + "lstm_headOuput_122k.npy"
        self.imageArray = None


    def loadLstmFile(self):
        imageArray = np.load(self.lstmDataFile)
        self.imageArray = imageArray

        print(imageArray)
        print(len(imageArray))
        print(imageArray.shape)

    def loadNpy(self, file):
        arr = np.load(file)

        print(arr)
        print(arr.shape)

if __name__ == '__main__':
    loader = loadNpy()
    # loader.loadLstmFile()
    print("Cell Output")
    loader.loadNpy(loader.lstmCellOutput)
    print("Heading Output")
    loader.loadNpy(loader.lstmHeadOutput)
