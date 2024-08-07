#!/usr/bin/env python3.5

"""--------------------------------------------------------------------------------
cnn_heading_model_2019.py

Author: Jinyoung Lim, Avik Bosshardt, Angel Sylvester and Maddie AlQatami

Updated: Summer 2022 by Bea Bautista, Yifan Wu, and Shoske Noma

This file can build and train CNN and load checkpoints/models for predicting headings using cell input.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.
Adapted from olin_cnn.py

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes


--------------------------------------------------------------------------------"""

import os
import numpy as np
from tensorflow import keras
import cv2
import time
from src.match_seeker.scripts.olri_classifier.paths import DATA, frames, checkPts
from src.match_seeker.scripts.olri_classifier.imageFileUtils import makeFilename
from src.match_seeker.scripts.olri_classifier.frameCellMap import FrameCellMap
import random

### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

class HeadingPredictModel(object):
    def __init__(self, eval_ratio=11.0/61.0, loaded_checkpoint=None, dataImg=None, dataLabel= None, outputSize= 8,
                 image_size=100, image_depth=2, data_name = None, testData = None, testFrames = None, checkPtsDirectory = None):
        ### Set up paths and basic model hyperparameters

        self.checkpoint_dir = DATA + "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.dataImg = dataImg
        self.dataLabel = dataLabel
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_eval = None
        self.train_images = None
        self.train_labels = None
        self.eval_images = None
        self.eval_labels = None
        self.data_name = data_name
        self.loss = keras.losses.categorical_crossentropy
        self.testData = testData
        self.loaded_checkpoint = loaded_checkpoint
        self.frames = testFrames
        self.mean_image = self.testData + "TRAININGDATA_100_500_mean.npy"
        self.meanData = np.load(self.mean_image)
        self.frameIDtext = self.testData + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        if loaded_checkpoint:
            self.loaded_checkpoint = loaded_checkpoint
            self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
            self.model.load_weights(self.loaded_checkpoint)
        else:
            # Code for cell input
            self.model = self.build_cnn()  # CNN

        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])

    def loadData(self):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values."""
        self.image = np.load(self.dataImg)
        self.label = np.load(self.dataLabel)
        self.image_totalImgs = self.image.shape[0]

        try:
            self.image_depth = self.image[0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int((self.eval_ratio * self.image_totalImgs))
        print("This is the total images", self.image_totalImgs)
        print("This is the ratio", self.num_eval)


        np.random.seed(2845) #45600

        if (len(self.image) == len(self.label)):
            p = np.random.permutation(len(self.image))
            self.image = self.image[p]
            self.label = self.label[p]
        else:
            print("Image data and heading data are  not the same size")
            return 0

        self.train_images = self.image[:-self.num_eval, :]
        print("This is the len of train images after it has been divided", len(self.train_images))
        self.eval_images = self.image[-self.num_eval:, :]

        print("THIS IS THE TOTAL SIZE BEFORE DIVIDING THE DATA", len(self.label))
        self.train_labels = self.label[:-self.num_eval, :]
        print("This is cutting the labels!!!!!", len(self.train_labels))
        self.eval_labels = self.label[-self.num_eval:, :]



    def train(self):
        """Sets up the loss function and optimizer, and then trains the model on the current training data. Quits if no
        training data is set up yet."""
        print("This is the shape of the train images!!", self.train_images.shape)
        if self.train_images is None:
            print("No training data loaded yet.")
            return 0

        self.model.fit(
            self.train_images, self.train_labels,
            batch_size= 1,
            epochs=20,
            verbose=1,
            validation_data=(self.eval_images, self.eval_labels),
            shuffle=True,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    batch_size=1,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )


    def build_cnn(self):
        """Builds a network that takes an image and an extra channel for the cell number, and produces the heading."""
        print("Building a model that takes cell number as input")
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last",
            input_shape=[self.image_size, self.image_size, self.image_depth]
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last"
        ))
        model.add(keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same",
        ))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dense(units=256, activation="relu"))

        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense(units=self.outputSize, activation="softmax"))
        model.summary()
        return model


    # TESTING CODE

    def cleanImage(self, image, mean=None, imageSize=100):
        """Preprocessing the images in similar ways to the training dataset of 2019 model."""
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
        meaned = np.subtract(grayed, mean)
        return shrunkenIm, grayed, meaned

    def cleanImageRandomCrop(self, image, mean=None, imageSize=100):
        """Alternative preprocessing function to cleanImage that crops the input image to a 100x100 image starting
        at a random x and y position. Resulted in very bad performance, we might not use anymore."""
        image = cv2.resize(image, (170, 128))
        x = random.randrange(0, 70)
        y = random.randrange(0, 28)
        cropped_image = image[y:y + imageSize, x:x + imageSize]
        grayed = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        meaned = np.subtract(grayed, mean)
        return cropped_image, grayed, meaned

    def test(self, n, randomChoose = True, randomCrop = False):
        """This runs each of the first n images in the folder of frames through the heading-output network, reporting how often the correct
        heading was produced, and how often the correct heading was in the top 3 and top 5."""
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        print("Setting up preprocessor to get frames data...")
        dPreproc = FrameCellMap(dataFile=self.frameIDtext)
        countPerfect = 0
        countTop3 = 0
        countTop5 = 0
        if randomChoose:
            iterations = n
        else:
            iterations = 1
        for i in range(iterations):
            if randomChoose:
                index = random.randrange(95000) - 1
            else:
                index = n - 1
            print("===========", index)
            imFile = makeFilename(frames, index)
            imageB = cv2.imread(imFile)
            if imageB is None:
                print(" image not found")
                continue
            cellB = dPreproc.frameData[index]['cell']
            headingB = dPreproc.frameData[index]['heading']
            headingIndex = potentialHeadings.index(
                headingB)  # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
            predB, output = self.predictSingleImageAllData(imageB, cellB, randomCrop=randomCrop)
            topThreePercs, topThreeHeadings = self.findTopX(3, output)
            topFivePercs, topFiveHeadings = self.findTopX(5, output)
            print("headingIndex =", headingIndex, "   predB =", predB)
            print("Top three:", topThreeHeadings, topThreePercs)
            print("Top five:", topFiveHeadings, topFivePercs)
            if predB == headingIndex:
                countPerfect += 1
            if headingIndex in topThreeHeadings:
                countTop3 += 1
            if headingIndex in topFiveHeadings:
                countTop5 += 1
            x = cv2.waitKey(50)
            if chr(x & 0xFF) == 'q':
                break
        print("Count of perfect:", countPerfect)
        print("Count of top 3:", countTop3)
        print("Count of top 5:", countTop5)


    def findTopX(self, x, numList):
        """Given a number and a list of numbers, this finds the x largest values in the number list, and reports
        both the values, and their positions in the numList."""
        topVals = [0.0] * x
        topIndex = [None] * x
        for i in range(len(numList)):
            val = numList[i]

            for j in range(x):
                if topIndex[j] is None or val > topVals[j]:
                    break
            if val > topVals[x - 1]:
                topIndex.insert(j, i)
                topVals.insert(j, val)
                topIndex.pop(-1)
                topVals.pop(-1)
        return topVals, topIndex


    def predictSingleImageAllData(self, image, cell, randomCrop = False):
        """Given an image converts it to be suitable for the network, runs the model and returns
        the resulting prediction."""
        if randomCrop:
            smallerB, grayB, processedB = self.cleanImageRandomCrop(image, self.meanData)
        else:
            smallerB, grayB, processedB = self.cleanImage(image, self.meanData)
        cellBArr = cell * np.ones((100, 100, 1))
        cleanImage = np.concatenate((np.expand_dims(processedB, axis=-1), cellBArr), axis=-1)
        # dispProcB = cv2.convertScaleAbs(processedB)
        # cv2.imshow("Image B", cv2.resize(image, (400, 400)))
        # cv2.moveWindow("Image B", 50, 50)
        # cv2.imshow("Smaller B", cv2.resize(smallerB, (400, 400)))
        # cv2.moveWindow("Smaller B", 50, 500)
        # cv2.imshow("Gray B", cv2.resize(grayB, (400, 400)))
        # cv2.moveWindow("Gray B", 500, 500)
        # cv2.imshow("Proce B", cv2.resize(dispProcB, (400, 400)))
        # cv2.moveWindow("Proce B", 500, 50)
        listed = np.array([cleanImage])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
        print("Model predict shape:", modelPredict.shape, "Model predicts:", modelPredict)
        print("predict[0] shape:", modelPredict[0].shape, "predict[0]:", modelPredict[0])
        return maxIndex, modelPredict[0]

# def loading_bar(start,end, size = 20):
#     # Useful when running a method that takes a long time
#     loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
#     if start % 10 == 0:
#         print(loadstr)
#
#
# def check_data():
#     data = np.load(DATA + 'IMG_CellInput_12K.npy')
#     np.random.shuffle(data)
#     print(data[0])
#     potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#     for i in range(len(data)):
#         print("cell:"+str(np.argmax(data[i][1])))
#         print("heading:"+str(potentialHeadings[int(data[i][0][0,0,1])]))
#         cv2.imshow('im',data[i][0][:,:,0])
#         cv2.moveWindow('im',200,200)
#         cv2.waitKey(0)


if __name__ == "__main__":
    # check_data()
    headingPredictor = HeadingPredictModel(
        # dataImg= DATA + 'IMG_CellInput_12K.npy',
        # dataLabel = DATA + 'Heading_CellInput12k.npy',
        # data_name = "testheadingpredictor"

        #for testing
        loaded_checkpoint=checkPts + "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5",
        testData = DATA, testFrames = frames
    )

    # print("Classifier built")
    # headingPredictor.loadData()
    # print("Data loaded")
    # headingPredictor.train()

    #print("Tests for Heading Predictor")
    # print("Randomly Cropped Images")
    # headingPredictor.test(1000, randomCrop=True)

    print("Original Test Call")
    headingPredictor.test(100, randomCrop=False)

