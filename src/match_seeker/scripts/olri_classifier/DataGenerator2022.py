import numpy as np
from paths import DATA, frames
from tensorflow import keras
import cv2
import os
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap

"""
Updated Data Generator that preprocesses images into desired input form
and constructs batches of tuples of images and corresponding labels that allows the model to
not load in the entire .npy file or the whold dataset for training or validation

Created Summer 2022
Authors: Bea Bautista, Yifan Wu, Shosuke Noma
"""


class DataGenerator2022(keras.utils.Sequence):
    def __init__(self, frames = frames, batch_size=20, shuffle=True,
                 img_size = 224, testData = DATA, seed = 25,
                 train = True, eval_ratio=11.0/61.0, generateForCellPred = True, cellPredWithHeadingIn = False):

        self.batch_size = batch_size
        self.frameIDtext = testData + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        self.shuffle = shuffle
        self.img_size = img_size
        self.image_path = frames
        np.random.seed(seed) #setting the random seed insures the training and testing data are not mix
        self.allImages = self.createListofPaths()
        self.eval_ratio = eval_ratio
        self.allImages, self.valImages = self.traintestsplit(self.allImages, self.eval_ratio)
        self.labelMap = None
        self.generateForCellPred = generateForCellPred
        self.cellPredWithHeadingIn = cellPredWithHeadingIn
        self.potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
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

    def cleanImage(self, image):
        """Preprocessing the images into the correct input form."""
        shrunkenIm = cv2.resize(image, (self.img_size, self.img_size))
        processedIm = shrunkenIm / 255.0
        return processedIm

    def __data_generation(self, list_frame_temp):
        'Generates data containing batch_size images'

        self.labelMap = FrameCellMap(dataFile = self.frameIDtext)

        # # Initialization
        # X = np.empty((self.batch_size)) #IS AN ARRAY WITHOUT INITIALIZING THE ENTRIES OF SHAPE (20, 100, 100, 1, 1)
        # Y = np.empty((self.batch_size), dtype=int)

        X = [None] * self.batch_size
        Y = [None] * self.batch_size

        # Generate data
        if self.generateForCellPred:
            for i, filename in enumerate(list_frame_temp):
                frameNum = extractNum(filename)
                raw = cv2.imread(self.image_path + filename)
                cell = self.labelMap.frameData[frameNum]['cell']

                if self.cellPredWithHeadingIn:
                    im = self.cleanImage(raw)
                    cell_arr = cell * np.ones((im.shape[0], im.shape[1], 1))
                    X[i] = np.concatenate((np.expand_dims(im, axis=-1), cell_arr), axis=-1)
                else:
                    X[i] = self.cleanImage(raw)
                Y[i] = cell
                print(X[i].shape)
        else:
            for i, filename in enumerate(list_frame_temp):
                frameNum = extractNum(filename)
                raw = cv2.imread(self.image_path + filename)
                X[i] = self.cleanImage(raw)
                heading = self.labelMap.frameData[frameNum]['heading']
                headingIndex = self.potentialHeadings.index(heading)
                if headingIndex == 8: #the 0th index is 0 degree and is the same as the 8th index 360 degrees
                    headingIndex = 0
                Y[i] = headingIndex
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


