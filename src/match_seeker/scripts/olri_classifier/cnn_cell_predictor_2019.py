"""--------------------------------------------------------------------------------
cnn_cell_predictor_2019.py

Updated: Summer 2022

This file can build and train CNN and load checkpoints/models for predicting cells.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""

import os
import numpy as np
from tensorflow import keras
import cv2
import time
from paths import DATA, frames, checkPts
from imageFileUtils import makeFilename
import random
from frameCellMap import FrameCellMap

### Uncomment next line to use CPU instead of GPU: ###
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CellPredictor2019(object):
    def __init__(self, eval_ratio=11.0/61.0, loaded_checkpoint=None, dataImg=None, dataLabel= None, outputSize= 271,
                 image_size=100, image_depth=2, data_name = None, testData = DATA, testFrames = frames, checkPtsDirectory = checkPts):
        ### Set up paths and basic model hyperparameters

        self.saving_checkpoint_dir = checkPts+"olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
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
            self.model = self.cnn_headings()  # CNN

        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])


    def loadData(self):
        """Loads the data from the given data file, setting several instance variables to hold training and testing
        inputs and outputs, as well as other helpful values."""

        self.image = np.load(self.dataImg)
        # self.image = self.image[:,:,:,0] #OBTAINING JUST THE IMAGE
        # self.image = self.image.reshape(len(self.image), 100, 100, 1) #WHEN DOING IMAGE ALONE
        self.image_totalImgs = self.image.shape[0]
        self.label = np.load(self.dataLabel)

        try:
            self.image_depth = self.image[0].shape[2]
        except IndexError:
            self.image_depth = 1

        self.num_eval = int((self.eval_ratio * self.image_totalImgs))
        print("This is the total images", self.image_totalImgs)
        print("This is the ratio", self.num_eval)

        np.random.seed(2845) #45600 #shuffles the orders of the photos
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
                    self.saving_checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    period=1  # save every n epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.saving_checkpoint_dir,
                    batch_size=1,
                    write_images=False,
                    write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )


    def cnn_headings(self):
        """Builds a network that takes an image and heading as extra channel along with image and produces the cell number."""

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
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=self.outputSize, activation="softmax"))
        model.summary()
        return model

    def predictSingleImageAllData(self, image, heading, randomCrop=False):
        """Given an image converts it to be suitable for the network, runs the model and returns
        the resulting prediction."""
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        headingIndex = potentialHeadings.index(heading)  #converting from 0, 45, 90, etc. to 0, 1, 2, etc
        if randomCrop:
            smallerB, grayB, processedB = self.cleanImageRandomCrop(image, self.meanData)
        else:
            smallerB, grayB, processedB = self.cleanImage(image, self.meanData)
        headBArr = headingIndex * np.ones((100, 100, 1))
        cleanImage = np.concatenate((np.expand_dims(processedB, axis=-1), headBArr), axis=-1)
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

    def test(self, n, randomChoose = True, randomCrop = False):
        """This runs each of the random n images in the folder of frames through the cell-output network, reporting how
        often the correct cell was produced0, and how often the correct heading was in the top 3 and top 5.
        Or when setting random to be false, it tests the model on the n-th image."""

        print("Setting up preprocessor to get frame data...")
        dPreproc = FrameCellMap(dataFile=self.frameIDtext)
        countPerfect = 0
        countTop3 = 0
        countTop5 = 0
        if randomChoose:
            iterations = n
        else: iterations = 1
        for i in range(iterations):
            if randomChoose:
                index = random.randrange(95000)-1
            else: index = n - 1
            print("===========", index)
            imFile = makeFilename(self.frames, index)
            imageB = cv2.imread(imFile)
            if imageB is None:
                print(" image not found")
                continue
            cellB = dPreproc.frameData[index]['cell']
            headingB = dPreproc.frameData[index]['heading']
            predB, output = self.predictSingleImageAllData(imageB, headingB, randomCrop=randomCrop)
            topThreePercs, topThreeCells = self.findTopX(3, output)
            topFivePercs, topFiveCells = self.findTopX(5, output)
            print("cellB =", cellB, "   predB =", predB)
            print("Top three:", topThreeCells, topThreePercs)
            print("Top five:", topFiveCells, topFivePercs)
            if predB == cellB:
                countPerfect += 1
            if cellB in topThreeCells:
                countTop3 += 1
            if cellB in topFiveCells:
                countTop5 += 1
            x = cv2.waitKey(50)
            if chr(x & 0xFF) == 'q':
                break
        print("Count of perfect:", countPerfect)
        print("Count of top 3:", countTop3)
        print("Count of top 5:", countTop5)

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

def loading_bar(start,end, size = 20):
    # Useful when running a method that takes a long time
    loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
    if start % 10 == 0:
        print(loadstr)


# def check_data():
#     data = np.load(DATA + 'TRAININGDATA_100_500_heading-input_gnrs.npy')
#     print(data.shape)
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
    cellPredictor = CellPredictor2019(
        #for setting up for building and training model
        # dataImg= DATA +"Img_w_head_13k.npy",
        # dataLabel = DATA + 'cell_ouput13k.npy',
        # data_name = "testCellPredictor"

        #for testing existing model
        loaded_checkpoint = checkPts + "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
        testData = DATA, testFrames = frames
    )
    # print("Classifier built")
    # cellPredictor.loadData()
    # print("Data loaded")
    # cellPredictor.train()

    print("Tests the cell predictor")
    cellPredictor.test(100)
