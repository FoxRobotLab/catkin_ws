"""--------------------------------------------------------------------------------
cnn_cell_predictor_RGBinput.py

Updated: Summer 2022

This file can build and train CNN and load checkpoints/models for predicting cells.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""

import os
import random
import time

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
# from sklearn.model_selection import train_test_split

from paths import DATA
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap

print(tf.__version__)

### Uncomment next line to use CPU instead of GPU: ###
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CellPredictor2019(object):
    def __init__(self, checkPts, eval_ratio=11.0/61.0, loaded_checkpoint=None, imagesFolder=None, imagesParent=None,
                 labelMapFile=None, outputSize=271, image_size=100, image_depth=3, data_name=None, dataSize = 0,
                 testData=DATA, batch_size = 32, seed=123456):
        ### Set up paths and basic model hyperparameters
        # TODO: We need to add a comment here that describes exactly what each of these inputs means!!
        self.checkpoint_dir = checkPts + "olin_cnn_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.image_size = image_size
        self.image_depth = image_depth
        self.batch_size = batch_size
        self.seed = seed
        self.num_eval = None
        self.train_images = None
        self.train_labels = None
        self.eval_images = None
        self.eval_labels = None
        self.data_name = data_name
        self.dataSize = dataSize
        self.testData = testData
        self.loaded_checkpoint = loaded_checkpoint
        self.mean_image = self.testData + "TRAININGDATA_100_500_mean.npy"
        self.frameIDtext = self.testData + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        self.frames = imagesFolder
        self.framesParent = imagesParent
        self.labelMapFile = labelMapFile
        self.labelMap = None
        self.train_ds = None
        self.val_ds = None
        if loaded_checkpoint:
            self.loaded_checkpoint = checkPts + loaded_checkpoint

    def prepDatasets(self):
        """Finds the cell labels associated with the files in the frames folder, and then sets up two
        data generators to produce the data in batches."""
        self.labelMap = FrameCellMap(imageDir=self.frames, dataFile=self.labelMapFile)  # Susan added this
        files = [f for f in os.listdir(self.frames) if f.endswith("jpg")]
        cellLabels = [self.labelMap.frameData[fNum]['cell'] for fNum in map(extractNum, files)]
        self.train_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels, subset="training",
                                                                 validation_split=0.2,  seed=self.seed,
                                                                 image_size=(self.image_size, self.image_size),
                                                                 batch_size=self.batch_size)
        self.val_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels,subset="validation",
                                                               validation_split=0.2, seed=self.seed,
                                                               image_size=(self.image_size, self.image_size),
                                                               batch_size=self.batch_size)



    def buildNetwork(self):
        """Builds the network, saving it to self.model."""
        if self.loaded_checkpoint:
            self.model = keras.models.load_model(self.loaded_checkpoint) #, compile=True)
            self.model.load_weights(self.loaded_checkpoint)
        else:
            self.model = self.cnn()  # CNN

        self.model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
                           metrics=["accuracy"])


    def train(self):
        """Sets up the loss function and optimizer, and then trains the model on the current training data. Quits if no
        training data is set up yet."""

        self.model.fit(
            self.train_ds,
            batch_size= self.batch_size,
            epochs=3,
            verbose=1,
            validation_data=self.val_ds,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name +".hdf5",
                    save_freq=200,
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    write_images=False,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )


    def cnn(self):
        """Builds a network that takes an image and produces the cell number."""

        model = keras.models.Sequential()
        # model.add(keras.layers.Rescaling(scale = 1./255, offset=0.0, **kwargs))
        # model.add(keras.layers.Resizing(
        #         self.image_size, self.image_size, interpolation="bilinear", crop_to_aspect_ratio=False, **kwargs))

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


    def predictSingleImageAllData(self, cleanImage):
        """Given a "clean" image that has been converted to be suitable for the network, this runs the model and returns
        the resulting prediction."""
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

    # def test(self, n, randomChoose = True, randomCrop = False):
    #     """This runs each of the random n images in the folder of frames through the cell-output network, reporting how
    #     often the correct cell was produced0, and how often the correct heading was in the top 3 and top 5.
    #     Or when setting random to be false, it tests the model on the n-th image."""
    #
    #     potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    #     mean = np.load(self.mean_image)
    #     print("Setting up preprocessor to get frame data...")
    #     dPreproc = FrameCellMap(dataFile=self.frameIDtext)
    #     countPerfect = 0
    #     countTop3 = 0
    #     countTop5 = 0
    #     if randomChoose:
    #         iterations = n
    #     else: iterations = 1
    #     for i in range(iterations):
    #         if randomChoose:
    #             index = random.randrange(95000)-1
    #         else: index = n - 1
    #         print("===========", index)
    #         imFile = makeFilename(self.frames, index)
    #         imageB = cv2.imread(imFile)
    #         if imageB is None:
    #             print(" image not found")
    #             continue
    #         cellB = dPreproc.frameData[index]['cell']
    #         headingB = dPreproc.frameData[index]['heading']
    #         headingIndex = potentialHeadings.index(headingB)  #converting from 0, 45, 90, etc. to 0, 1, 2, etc.
    #
    #         if randomCrop:
    #             smallerB, grayB, processedB = self.cleanImageRandomCrop(imageB, mean)
    #         else:
    #             smallerB, grayB, processedB = self.cleanImage(imageB, mean)
    #         headBArr = headingIndex * np.ones((100, 100, 1))
    #         procBPlus = np.concatenate((np.expand_dims(processedB, axis=-1), headBArr), axis=-1)
    #         predB, output = self.predictSingleImageAllData(procBPlus)
    #         topThreePercs, topThreeCells = self.findTopX(3, output)
    #         topFivePercs, topFiveCells = self.findTopX(5, output)
    #         print("cellB =", cellB, "   predB =", predB)
    #         print("Top three:", topThreeCells, topThreePercs)
    #         print("Top five:", topFiveCells, topFivePercs)
    #         if predB == cellB:
    #             countPerfect += 1
    #         if cellB in topThreeCells:
    #             countTop3 += 1
    #         if cellB in topFiveCells:
    #             countTop5 += 1
    #         dispProcB = cv2.convertScaleAbs(processedB)
    #         cv2.imshow("Image B", cv2.resize(imageB, (400, 400)))
    #         cv2.moveWindow("Image B", 50, 50)
    #         cv2.imshow("Smaller B", cv2.resize(smallerB, (400, 400)))
    #         cv2.moveWindow("Smaller B", 50, 500)
    #         cv2.imshow("Gray B", cv2.resize(grayB, (400, 400)))
    #         cv2.moveWindow("Gray B", 500, 500)
    #         cv2.imshow("Proce B", cv2.resize(dispProcB, (400, 400)))
    #         cv2.moveWindow("Proce B", 500, 50)
    #         x = cv2.waitKey(50)
    #         if chr(x & 0xFF) == 'q':
    #             break
    #     print("Count of perfect:", countPerfect)
    #     print("Count of top 3:", countTop3)
    #     print("Count of top 5:", countTop5)
    #
    # def cleanImage(self, image, mean=None, imageSize=100):
    #     """Preprocessing the images in similar ways to the training dataset of 2019 model."""
    #     shrunkenIm = cv2.resize(image, (imageSize, imageSize))
    #     grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
    #     meaned = np.subtract(grayed, mean)
    #     return shrunkenIm, grayed, meaned
    #
    # def cleanImageRandomCrop(self, image, mean=None, imageSize=100):
    #     """Alternative preprocessing function to cleanImage that crops the input image to a 100x100 image starting
    #     at a random x and y position. Resulted in very bad performance, we might not use anymore."""
    #     image = cv2.resize(image, (170, 128))
    #     x = random.randrange(0, 70)
    #     y = random.randrange(0, 28)
    #     cropped_image = image[y:y + imageSize, x:x + imageSize]
    #     grayed = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    #     meaned = np.subtract(grayed, mean)
    #     return cropped_image, grayed, meaned

# def loading_bar(start,end, size = 20):
#     # Useful when running a method that takes a long time
#     loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
#     if start % 10 == 0:
#         print(loadstr)


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
        # for setting up for building and training model
        eval_ratio=0.2,
        checkPts=DATA + "CHECKPOINTS/",
        data_name="CellPredictTest",
        imagesFolder=DATA + "frames/moreFrames/",
        imagesParent=DATA + "frames/",
        labelMapFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
        loaded_checkpoint=False,
        dataSize=95810,
        batch_size=32,
        seed=48823,
        # data_name = "testCellPredictor"
    )
    print("Classifier built")
    cellPredictor.prepDatasets()
    cellPredictor.buildNetwork()
    print(cellPredictor.checkpoint_dir)
    cellPredictor.train()

    # print("Tests the cell predictor")
    # cellPredictor.test(1000, randomChoose = False, randomCrop = True)
