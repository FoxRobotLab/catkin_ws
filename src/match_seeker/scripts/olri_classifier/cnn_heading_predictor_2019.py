#!/usr/bin/env python3.5

"""--------------------------------------------------------------------------------
cnn_heading_predictor_2019.py

Author: Jinyoung Lim, Avik Bosshardt, Angel Sylvester and Maddie AlQatami

Updated: Summer 2022 by Bea Bautista, Yifan Wu, and Shoske Noma

This file can build and train CNN and load checkpoints/models for predicting headings using cell input.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.
Adapted from olin_cnn.py

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes


--------------------------------------------------------------------------------"""


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os
import numpy as np
from tensorflow import keras
import cv2
import time
from paths import DATA, frames, pathToMatchSeeker
from imageFileUtils import makeFilename
from preprocessData import DataPreprocess

# ORIG import olin_inputs_2019 as oi2
import random
from cnn_lstm_functions import creatingSequence, getCorrectLabels, transfer_lstm_cellPred, CNN, transfer_lstm_headPred

### Uncomment next line to use CPU instead of GPU: ###
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class HeadingPredictor(object):
    def __init__(self, eval_ratio=11.0/61.0, checkpoint_name=None, dataImg=None, dataLabel= None, outputSize= None, model2020 = False,
                 image_size=224, image_depth=2, data_name = None):
        ### Set up paths and basic model hyperparameters

        self.checkpoint_dir = DATA + "CHECKPOINTS/olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001

        # self.model2020 = model2020

        self.dataImg = dataImg
        self.dataLabel = dataLabel
        self.dataArray = None
        self.image_size = image_size
        self.image_depth = image_depth
        self.num_eval = None
        self.train_images = None
        self.train_labels = None
        self.eval_images = None
        self.eval_labels = None
        self.data_name = data_name

        #Code for cell input
        self.model = self.build_cnn()  #CNN
        self.loss = keras.losses.categorical_crossentropy


        # elif self.model2020 == "Heading":  # no compiling for 2020 models
        #     self.loss = keras.losses.categorical_crossentropy
        #     self.model = keras.models.load_model(
        #         DATA + "CHECKPOINTS/olin_cnn_checkpoint-0720202216/CNN_headPred_all244Cell-01-0.27.hdf5")
        #     self.model.load_weights(
        #         DATA + "CHECKPOINTS/olin_cnn_checkpoint-0720202216/CNN_headPred_all244Cell-01-0.27.hdf5")hdf5


        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])

        # self.checkpoint_name = checkpoint_name
        # if self.checkpoint_name is not None:
        #     self.model.load_weights(self.checkpoint_name)


    def loadData(self):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values."""


        #ORIG self.dataArray = np.load(self.dataFile, allow_pickle=True, encoding='latin1')
        self.image = np.load(self.dataImg)
        #self.image = self.image[:,:,:,0] # #WHEN DOING IMAGE ALONE
        self.image = self.image.reshape(len(self.image), 100, 100, 1) #WHEN DOING IMAGE ALONE

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

        # if (self.checkpoint_name is None):
        #     self.model.compile(
        #         loss=self.loss,
        #         optimizer=keras.optimizers.SGD(lr=self.learning_rate),
        #         metrics=["accuracy"]
        #     )= self.label[



        #UNCOMMENT FOR OVERLAPPING
        ####################################################################
        # timeStepsEach = 400
        # self.train_images= creatingSequence(self.train_images, 400, 100)
        # timeSteps = len(self.train_images)
        # subSequences = int(timeSteps/timeStepsEach)
        # self.train_images = self.train_images.reshape(subSequences,timeStepsEach, 100, 100, 1)
        # self.train_labels = getCorrectLabels(self.train_labels, 400, 100)

        #
        # self.eval_images = creatingSequence(self.eval_images, 400, 100)
        # timeSteps = len(self.eval_images)
        # subSequences = int(timeSteps / timeStepsEach)
        # self.eval_images = self.eval_images.reshape(subSequences,timeStepsEach,100, 100, 1)
        # self.eval_labels = getCorrectLabels(self.eval_labels, 400, 100)

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

        # activate with softmax when training one label and sigmoid when training both headings and cells
        activation = "softmax"
        model.add(keras.layers.Dense(units=self.outputSize, activation=activation))
        model.summary()
        return model

    def cleanImage(self, image, mean=None, imageSize=100):
        """Preprocessing the images in similar ways to the training dataset of 2019 model."""
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
        meaned = np.subtract(grayed, mean)
        return shrunkenIm, grayed, meaned

    def testingOnHeadingOutputNetwork(self, n):
        """This runs each of the first n images in the folder of frames through the heading-output network, reporting how often the correct
        heading was produced, and how often the correct heading was in the top 3 and top 5."""
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        # cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
        cellOutputCheckpoint = "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5"
        meanFile = "TRAININGDATA_100_500_mean.npy"
        dataPath = DATA

        print("Setting up preprocessor to get frame data...")
        dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

        print("Loading mean...")
        mean = np.load(dataPath + meanFile)

        print("Setting up classifier loading checkpoints...")
        checkPts = dataPath + "CHECKPOINTS/"
        olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                         savedCheckpoint=checkPts + cellOutputCheckpoint,
                                         cellInput=True,
                                         outputSize=9,
                                         image_size=100,
                                         image_depth=2)
        countPerfect = 0
        countTop3 = 0
        countTop5 = 0
        for i in range(n):
            rand = random.randrange(95000)
            print("===========", rand)
            imFile = makeFilename(frames, rand)
            imageB = cv2.imread(imFile)
            if imageB is None:
                print(" image not found")
                continue
            cellB = dPreproc.frameData[rand]['cell']
            headingB = dPreproc.frameData[rand]['heading']
            headingIndex = potentialHeadings.index(
                headingB)  # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
            smallerB, grayB, processedB = self.cleanImage(imageB, mean)

            cellBArr = cellB * np.ones((100, 100, 1))
            procBPlus = np.concatenate((np.expand_dims(processedB, axis=-1), cellBArr), axis=-1)
            predB, output = olin_classifier.predictSingleImageAllData(procBPlus)
            topThreePercs, topThreeCells = self.findTopX(3, output)
            topFivePercs, topFiveCells = self.findTopX(5, output)
            print("headingIndex =", headingIndex, "   predB =", predB)
            print("Top three:", topThreeCells, topThreePercs)
            print("Top five:", topFiveCells, topFivePercs)
            if predB == headingIndex:
                countPerfect += 1
            if headingIndex in topThreeCells:
                countTop3 += 1
            if headingIndex in topFiveCells:
                countTop5 += 1
            dispProcB = cv2.convertScaleAbs(processedB)
            cv2.imshow("Image B", cv2.resize(imageB, (400, 400)))
            cv2.moveWindow("Image B", 50, 50)
            cv2.imshow("Smaller B", cv2.resize(smallerB, (400, 400)))
            cv2.moveWindow("Smaller B", 50, 500)
            cv2.imshow("Gray B", cv2.resize(grayB, (400, 400)))
            cv2.moveWindow("Gray B", 500, 500)
            cv2.imshow("Proce B", cv2.resize(dispProcB, (400, 400)))
            cv2.moveWindow("Proce B", 500, 50)
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

    # def getAccuracy(self):
    #     """Sets up the network, and produces an accuracy value on the evaluation data.
    #     If no data is set up, it quits."""
    #
    #     if self.eval_images is None:
    #         return
    #
    #     num_eval = 5000
    #     correctCells = 0
    #     correctHeadings = 0
    #     eval_copy = self.eval_images
    #     self.model.compile(loss=self.loss, optimizer=keras.optimizers.SGD(lr=0.001), metrics=["accuracy"])
    #     self.model.load_weights()
    #
    #     for i in range(num_eval):
    #         loading_bar(i,num_eval)
    #         image = eval_copy[i]
    #         image = np.array([image], dtype="float").reshape(-1, self.image_size, self.image_size, self.image_depth)
    #         potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    #
    #         pred = self.model.predict(image)
    #         print("correct:{}".format(np.argmax(self.eval_labels[i])))
    #         print("pred:{}".format(np.argmax(pred[0])))
    #         #cv2.imshow('im',image[0,:,:,0])
    #         #cv2.waitKey(0)
    #
    #         # print(np.argmax(labels[i][:self.num_cells]),np.argmax(pred[0][:self.num_cells]))
    #         # prinpredictt(np.argmax(labels[i][self.num_cells:]),np.argmax(pred[0][self.num_cells:]))
    #         # print(np.argmax(self),np.argmax(pred[0]))
    #         if np.argmax(self.eval_labels[i]) == np.argmax(pred[0]):
    #             correctCells += 1
    #         # if np.argmax(self.train_labels[i][self.num_cells-8:]) == np.argmax(pred[0][self.num_cells-8:]):
    #         #      correctHeadings += 1
    #
    #     print("%Correct Cells: " + str(float(correctCells) / num_eval))
    #     #print("%Correct Headings: " + str(float(correctHeadings) / num_eval))
    #     return float(correctCells) / num_eval


    # def retrain(self):
    #     """This method seems out of date, was used for transfer learning from VGG. DON"T CALL IT!"""
    #     # Use for retraining models included with keras
    #     # if training with headings cannot use categorical crossentropy to evaluate loss
    #     if self.checkpoint_name is None:
    #         self.model = keras.models.Sequential()
    #
    #         xc = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,
    #                                                     input_shape=(self.image_size, self.image_size, self.image_depth))
    #         for layer in xc.layers[:-1]:
    #             layer.trainable = False
    #
    #         self.model.add(xc)
    #         self.model.add(keras.layers.Flatten())
    #         self.model.add(keras.layers.Dropout(rate=0.4))
    #         # activate with softmax when training one label and sigmoid when training both headings and cells
    #         activation = self.train_with_headings*"sigmoid" + (not self.train_with_headings)*"softmax"
    #         self.model.add(keras.layers.Dense(units=self.outputSize, activation=activation))
    #         self.model.summary()
    #         self.model.compile(
    #             loss=self.loss,
    #             optimizer=keras.optimizers.Adam(lr=.001),
    #             metrics=["accuracy"]
    #         )
    #     else:
    #         print("Loaded model")
    #         self.model = keras.models.load_model(self.checkpoint_name, compile=False)
    #         self.model.compile(
    #             loss=self.loss,
    #             optimizer=keras.optimizers.Adam(lr=.001),
    #             metrics=["accuracy"]
    #         )
    #     print("Train:", self.train_images.shape, self.train_labels.shape)
    #     print("Eval:", self.eval_images.shape, self.eval_labels.shape)
    #     self.model.fit(
    #         self.train_images, self.train_labels,
    #         batch_size=100,
    #         epochs=10,
    #         verbose=1,
    #         validation_data=(self.eval_images, self.eval_labels),
    #         shuffle=True,
    #         callbacks=[
    #             keras.callbacks.History(),
    #             keras.callbacks.ModelCheckpoint(
    #                 self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
    #                 period=1  # save every n epoch
    #             )
    #             ,
    #             keras.callbacks.TensorBoard(
    #                 log_dir=self.checkpoint_dir,
    #                 batch_size=100,
    #                 write_images=False,
    #                 write_grads=True,
    #                 histogram_freq=0,
    #             ),
    #             keras.callbacks.TerminateOnNaN(),
    #         ]
    #     )


    # def precision(self,y_true, y_pred):
    #     """Precision metric.
    #
    #     Use precision in place of accuracy to evaluate models that have multiple outputs. Otherwise it's relatively
    #     unhelpful. The values returned during training do not represent the accuracy of the model. Use get_accuracy
    #     after training to evaluate models with multiple outputs.
    #
    #     Only computes a batch-wise average of precision.
    #
    #     Computes the precision, a metric for multi-label classification of how many selected items are relevant.
    #     """
    #     true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    #     predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    #     precision = true_positives / (predicted_positives + keras.backend.epsilon())
    #     return precision


    # def runSingleImage(self, num, input='heading'):
    #     imDirectory = DATA + 'frames/moreframes/'
    #     count = 0
    #     filename = makeFilename(imDirectory, num)
    #     # st = None
    #
    #     # for fname in os.listdir(imDirectory):
    #     #     if count == num:
    #     #         st = imDirectory + fname
    #     #         break
    #
    #     # print(imgs)
    #     # print(filename)
    #     if filename is not None:
    #         image = cv2.imread(filename)
    #         # print("This is image:", image)
    #         # print("This is the shape", image.shape)
    #         if image is not None:
    #             cellDirectory = DATA + 'frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt'
    #             count = 0
    #             with open(cellDirectory) as fp:
    #                 for line in fp:
    #                     (fNum, cell, x, y, head) = line.strip().split(' ')
    #                     if fNum == str(num):
    #                         break
    #                     count += 1
    #
    #
    #         # cell = oi2.getOneHotLabel(int(cell), 271)
    #         # cell_arr = []model.predict
    #         # im_arr = []
    #         # cell_arr.append(cell)
    #         # im_arr.append(image)
    #         #
    #         # cell_arr = np.asarray(cell_arr)
    #         # im_arr = np.asarray(im_arr)
    #
    #             if input=='heading':
    #                 image = clean_image(image, data='heading_channel', heading=int(head))
    #
    #             elif input=='cell':
    #                 image = clean_image(image, data='cell_channel', heading=int(cell))
    #
    #
    #
    #             return self.model.predict(image), cell
    #     return None


# def predictSingleImageAllData(self, cleanImage):
#     """Given a "clean" image that has been converted to be suitable for the network, this runs the model and returns
#     the resulting prediction."""
#     listed = np.array([cleanImage])
#     modelPredict = self.model.predict(listed)
#     maxIndex = np.argmax(modelPredict)
#     print("Model predicts:", modelPredict.shape, modelPredict)
#     print("predict[0]:", modelPredict[0].shape, modelPredict[0])
#     return maxIndex, modelPredict[0]
#
#
# def loading_bar(start,end, size = 20):
#     # Useful when running a method that takes a long time
#     loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
#     if start % 10 == 0:
#         print(loadstr)
#
#
# def check_data():
#     data = np.load(DATA + 'TRAININGDATA_100_500_heading-input_gnrs.npy')
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
    olin_classifier = HeadingPredictor(

        # dat aImg= DATA + 'SAMPLETRAININGDATA_IMG_withCellInput135K.npy',
        # dataLabel = DATA + 'SAMPLETRAININGDATA_HEADING_withCellInput135K.npy',
        #dataImg = DATA + 'lstm_Img_Cell_Input13k.npy',
        # dataImg= DATA +"Img_w_head_13k.npy",
        # dataImg=DATA + "lstm_Img_13k.npy",
        dataImg= DATA + 'Img_122k_ordered.npy',
        # dataLabel= DATA + 'lstm_cellOutput_122k.npy',
        dataLabel= DATA + 'lstm_headOuput_122k.npy',
        #dataLabel=DATA + 'lstm_head_13k.npy',
        # dataLabel = DATA + 'cell_ouput13k.npy',
        data_name = "CNN_cellPred_all244Cell_20epochs",
        outputSize= 271,
        eval_ratio= 11.0/61.0,
        image_size=100,
        model2020= True,
        image_depth= 1
    )
    print("Classifier built")
    olin_classifier.loadData()
    print("Data loaded")
    olin_classifier.train()




    # print(len(olin_classifier.train_images))
    #olin_classifier.train()
    # olin_classifier.getAccuracy()
    #ORIG count = 0
    # ORIG for i in range(1000):
    #     num = random.randint(0,95000)
    #     thing, cell = olin_classifier.runSingleImage(num)
    #     count += (np.argmax(thing)==cell)
    # print(count)


    # model = olin_classifier.threeConv()
    #olin_classifier.train()

    # self.cell_model = keras.models.load_model(
    #     "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
    #     compile=True)
