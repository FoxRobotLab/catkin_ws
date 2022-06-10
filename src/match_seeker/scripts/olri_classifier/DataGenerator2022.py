import numpy as np
from paths import DATA, frames
from tensorflow import keras
import random
import time
import cv2
import os
from imageFileUtils import makeFilename, extractNum
from preprocessData import DataPreprocess

""" 
Updated Data Generator that returns batches of images in an np array. Does not load in the entire .npy file
and instead uses a list of image paths to read images

Created Summer 2022
Authors: Bea Bautista, Yifan Wu, Shosuke Noma
"""


class DataGenerator2022(keras.utils.Sequence):
    def __init__(self, list_frames, batch_size=20,shuffle=True,
                 img_size = 100):
        self.list_frames = list_frames
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.mean = np.load(DATA + meanFile)
        #######
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.frameIDtext = self.testData + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        ########

        self.shuffle = shuffle
        self.img_size = img_size
        self.image_path = frames
        # self.on_epoch_end()

        self.createListofPaths()

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_frames) / self.batch_size))


    def createListofPaths(self):
        allFiles = os.listdir(self.image_path)
        allJpg = []
        for image in allFiles:
            if image.endswith('jpg'):
                allJpg.append(image)
        self.allImages = np.array(allJpg)
        np.random.shuffle(self.allImages)

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_frames))
    #
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    def __getitem__(self, index):
      'Generate one batch of data'

      # Find list of frames
      list_frame_temp = self.allImages[index*self.batch_size:(index+1)*self.batch_size]


      # Generate data
      X, Y = self.__data_generation(list_frame_temp)
      return X, Y

    def __data_generation(self, list_frame_temp):
        'Generates data containing batch_size images'

        dPreproc = DataPreprocess(dataFile=self.frameIDtext)

        # # Initialization
        # X = np.empty((self.batch_size)) #IS AN ARRAY WITHOUT INITIALIZING THE ENTRIES OF SHAPE (20, 100, 100, 1, 1)
        # Y = np.empty((self.batch_size), dtype=int)

        X = [None] * self.batch_size
        Y = [None] * self.batch_size

        # Generate data
        for i, filename in enumerate(list_frame_temp):
            frameNum = extractNum(filename)
            # Store sample
            X[i] = cv2.imread(self.image_path + filename) #Array of images
            Y[i] = dPreproc.frameData[frameNum]['cell']

        return np.array(X), np.array(Y) #Array of labels

if __name__ == "__main__":
    test = DataGenerator2022


