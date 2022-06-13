import numpy as np
from paths import DATA, frames
from tensorflow import keras
import random
import time
import cv2
import os
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap

""" 
Updated Data Generator that returns batches of images in an np array that allows the model to
not load in the entire .npy file and instead uses a list of image paths to read images in batches

Created Summer 2022
Authors: Bea Bautista, Yifan Wu, Shosuke Noma
"""


class DataGenerator2022(keras.utils.Sequence):
    def __init__(self, frames = frames, batch_size=20, shuffle=True,
                 img_size = 100, testData = DATA, seed = 25,
                 train = True, eval_ratio=11.0/61.0):

        self.batch_size = batch_size
        self.frameIDtext = testData + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        self.shuffle = shuffle
        self.img_size = img_size
        self.image_path = frames
        np.random.seed(seed)
        self.allImages = self.createListofPaths()
        self.eval_ratio = eval_ratio
        self.allImages, self.valImages = self.traintestsplit(self.allImages, self.eval_ratio)
        if not train:
            self.allImages = self.valImages

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.allImages) / self.batch_size))


    def createListofPaths(self):
        allFiles = os.listdir(self.image_path)
        allJpg = []
        for image in allFiles:
            if image.endswith('jpg'):
                allJpg.append(image)
        allImages = np.array(allJpg)
        np.random.shuffle(allImages)
        return allImages


    def on_epoch_end(self):
        'Reshuffles after each epoch'
        if self.shuffle:
            np.random.shuffle(self.allImages)


    def __getitem__(self, index):
      'Generate one batch of data'

      # Find list of frames
      list_frame_temp = self.allImages[index*self.batch_size:(index+1)*self.batch_size]


      # Generate data
      X, Y = self.__data_generation(list_frame_temp)
      return X, Y

    def cleanImage(self, image, imageSize=100):
        """Preprocessing the images in similar ways to the training dataset of 2019 model."""
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        processedIm = shrunkenIm / 255.0
        return processedIm

    def __data_generation(self, list_frame_temp):
        'Generates data containing batch_size images'

        dPreproc = FrameCellMap(dataFile=self.frameIDtext)

        # # Initialization
        # X = np.empty((self.batch_size)) #IS AN ARRAY WITHOUT INITIALIZING THE ENTRIES OF SHAPE (20, 100, 100, 1, 1)
        # Y = np.empty((self.batch_size), dtype=int)

        X = [None] * self.batch_size
        Y = [None] * self.batch_size

        # Generate data
        for i, filename in enumerate(list_frame_temp):
            frameNum = extractNum(filename)
            # Store sample
            raw = cv2.imread(self.image_path + filename) #Array of images
            X[i] = self.cleanImage(raw)
            Y[i] = dPreproc.frameData[frameNum]['cell']

        return np.array(X), np.array(Y) #Array of labels

    def traintestsplit(self, images, eval_ratio):
        '''Split the data passed in based on the evaluation ratio into
        training and testing datasets, assuming it's already randomized'''
        image_totalImgs = np.size(images)
        num_eval = int((eval_ratio * image_totalImgs))
        train_images = images[:-num_eval]
        eval_images = images[-num_eval:]
        return train_images, eval_images

# if __name__ == "__main__":
#     test = DataGenerator2022


