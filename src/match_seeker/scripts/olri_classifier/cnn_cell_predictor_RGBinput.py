"""--------------------------------------------------------------------------------
cnn_cell_predictor_RGBinput.py

Updated: Summer 2022

This file can build and train CNN and load checkpoints/models for predicting cells.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""

import time
import cv2
import tensorflow as tf

import os
import numpy as np
from tensorflow import keras
# from cv2 import imread, imshow, resize, moveWindow, waitKey
import time
from paths import DATA, checkPts, frames
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap
from DataGenerator2022 import DataGenerator2022
import random

### Uncomment next line to use CPU instead of GPU: ###
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CellPredictorRGB(object):
    def __init__(self, checkPointFolder = None, loaded_checkpoint = None, imagesFolder = None, imagesParent = None, labelMapFile = None, data_name=None,
                 eval_ratio=11.0 / 61.0, outputSize=271, image_size=100, image_depth=3, dataSize = 0, batch_size = 32, seed=123456):
        """
        :param checkPointFolder: Destination path where checkpoints should be saved
        :param loaded_checkpoint: Name of the last saved checkpoint file inside checkPointFolder; used to continue training or conduct tests
        :param imagesFolder: Path of the folder containing all 95k image files
        :param imagesParent: Path of the parent folder of the imagesFolder, required by prepDatasets which isn't yet working
        :param labelMapFile: Path of the txt file that contains the cell/heading/x y coordinate information for each of the 95k images
        :param data_name: Name that each checkpoint is saved under, indicates whether model is trained on all images or just a subset
        :param eval_ratio: Ratio of validation data as a proportion of the entire data, used for splitting testing and validation data
        :param outputSize: Number of output categories, 271 cells
        :param image_size: Target length of images after resizing, typically 100x100 pixels
        :param image_depth: Number of channels in the input images; 3 for RGB color images, 1 for black and white
        :param dataSize: NOT CURRENTLY USED - need to ask Susan
        :param batch_size: Number of images in each batch fed into the model via Data Generator pipeline
        :param seed: Random seed to ensure that random splitting remains the same between training and validation
        """
        ### Set up paths and basic model hyperparameters
        self.checkpoint_dir = checkPointFolder + "2022CellPredict_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.image_size = image_size
        self.image_depth = image_depth
        self.batch_size = batch_size
        self.seed = seed
        self.num_eval = None
        # self.train_images = None
        # self.train_labels = None
        # self.eval_images = None
        # self.eval_labels = None
        self.data_name = data_name
        self.dataSize = dataSize #len of ...?
        self.loaded_checkpoint = loaded_checkpoint
        self.frames = imagesFolder
        self.framesParent = imagesParent
        self.labelMapFile = labelMapFile
        self.labelMap = None
        self.train_ds = None
        self.val_ds = None
        if loaded_checkpoint:
            self.loaded_checkpoint = checkPointFolder + loaded_checkpoint

    def buildMap(self):
        """Builds dictionaries containing the corresponding cell, heading, and location information for each frame,
        saving it to self.labelMap."""
        self.labelMap = FrameCellMap(dataFile=self.labelMapFile)

    #not compatible with tensorflow 1
    # def prepDatasets(self):
    #     """Finds the cell labels associated with the files in the frames folder, and then sets up two
    #     data generators to produce the data in batches."""
    #     self.buidMap()
    #     files = [f for f in os.listdir(self.frames) if f.endswith("jpg")]
    #     cellLabels = [self.labelMap.frameData[fNum]['cell'] for fNum in map(extractNum, files)]
    #     self.train_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels, subset="training",
    #                                                              validation_split=0.2,  seed=self.seed,
    #                                                              image_size=(self.image_size, self.image_size),
    #                                                              batch_size=self.batch_size)
    #     self.train_ds = self.train_ds.map(lambda x, y: (x /255., y))
    #
    #     self.val_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels,subset="validation",
    #                                                            validation_split=0.2, seed=self.seed,
    #                                                            image_size=(self.image_size, self.image_size),
    #                                                            batch_size=self.batch_size)
    #     self.val_ds = self.val_ds.map(lambda x, y: (x / 255., y))

    def prepDatasets(self):
        """Finds the cell labels associated with the files in the frames folder, and then sets up two
        data generators to preprocess data and produce the data in batches."""
        self.train_ds = DataGenerator2022()
        self.val_ds = DataGenerator2022(train = False)

    def buildNetwork(self):
        """Builds the network, saving it to self.model."""
        if self.loaded_checkpoint:
            self.model = keras.models.load_model(self.loaded_checkpoint) #, compile=True)
            self.model.load_weights(self.loaded_checkpoint)
        else:
            self.model = self.cnn()  # CNN

        self.model.compile(
            loss= keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=["accuracy"])

    # def train(self, epochs = 20):
    #     """Sets up the loss function and optimizer, and then trains the model on the current training data. Quits if no
    #     training data is set up yet."""
    #
    #     print("This is the shape of the train images!!", self.train_images.shape)
    #     if self.train_images is None:
    #         print("No training data loaded yet.")
    #         return 0
    #
    #     self.model.fit(
    #         self.train_images, self.train_labels,
    #         batch_size= 1,
    #         epochs=epochs,
    #         verbose=1,
    #         validation_data=(self.eval_images, self.eval_labels),
    #         shuffle=True,
    #         callbacks=[
    #             keras.callbacks.History(),
    #             keras.callbacks.ModelCheckpoint(
    #                 self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
    #                 period=1  # save every n epoch
    #             ),
    #             keras.callbacks.TensorBoard(
    #                 log_dir=self.checkpoint_dir,
    #                 batch_size=1,
    #                 write_images=False,
    #                 write_grads=True,
    #                 histogram_freq=1,
    #             ),
    #             keras.callbacks.TerminateOnNaN()
    #         ]
    #     )


    def train_withGenerator(self, training_generator, validation_generator, epoch = 5 ):
        self.model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            workers=6,
                            #steps_per_epoch = 6100, #Sample data ---> 6
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
                                    write_grads=True
                                ),
                                keras.callbacks.TerminateOnNaN()
                ],
                            epochs= epoch)




    def cnn(self):
        """Builds a network that takes an image and produces the cell number."""

        model = keras.models.Sequential()

        #rescale and resize layers not compatible with tensorflow 1, so they're done in prepDatasets function.
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

    def test(self, n, randomChoose = True):
        """This runs each of the random n images in the folder of frames through the cell-output network, reporting how
        often the correct cell was produced0, and how often the correct heading was in the top 3 and top 5.
        Or when setting randomChoose to be false, it tests the model on the n-th image."""

        self.buildMap()
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
            print("===========", index+1)
            imFile = makeFilename(self.frames, index)
            print(imFile)
            image = cv2.imread(imFile)
            if image is None:
                print(" image not found")
                continue
            cell = self.labelMap.frameData[index]['cell']

            processed = self.cleanImage(image)
            pred, output = self.predictSingleImageAllData(processed)
            topThreePercs, topThreeCells = self.findTopX(3, output)
            topFivePercs, topFiveCells = self.findTopX(5, output)
            print("cellB =", cell, "   pred =", pred)
            print("Top three:", topThreeCells, topThreePercs)
            print("Top five:", topFiveCells, topFivePercs)
            if pred == cell:
                countPerfect += 1
            if cell in topThreeCells:
                countTop3 += 1
            if cell in topFiveCells:
                countTop5 += 1
            cv2.imshow("Image B", cv2.resize(image, (400, 400)))
            cv2.moveWindow("Image B", 50, 50)
            x = cv2.waitKey(50)
            if chr(x & 0xFF) == 'q':
                break
        print("Count of perfect:", countPerfect)
        print("Count of top 3:", countTop3)
        print("Count of top 5:", countTop5)


    def cleanImage(self, image, imageSize=100):
        """Process a single image into the correct input form for 2020 model, mainly used for testing."""
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        processedIm = shrunkenIm / 255.0
        return processedIm

# def loading_bar(start,end, size = 20):
#     # Useful when running a method that takes a long time
#     loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
#     if start % 10 == 0:
#         print(loadstr)

if __name__ == "__main__":
    #check_data()
    cellPredictor = CellPredictorRGB(
        # dataSize=95810,
        data_name="FullData",
        checkPointFolder=checkPts,
        imagesFolder=frames,
        #imagesParent=DATA + "frames/",
        labelMapFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
        loaded_checkpoint="2022CellPredict_checkpoint-0613221515/FullData-05-0.48.hdf5",
    )

    cellPredictor.buildNetwork()

    #for training
    #cellPredictor.prepDatasets()
    #cellPredictor.train_withGenerator(cellPredictor.train_ds, cellPredictor.val_ds)

    #for Tensorflow 1
    # training_generator = DataGenerator2022()
    # validation_generator = DataGenerator2022(train = False)
    # cellPredictor.train_withGenerator(training_generator,validation_generator)

    #for testing
    cellPredictor.test(100)
