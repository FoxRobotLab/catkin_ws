"""--------------------------------------------------------------------------------
cnn_heading_predictor_RGBinput.py

Updated: Summer 2022

This file can build and train CNN and load checkpoints/models for predicting headings.
The model was originally created in 2019 that takes a picture and its cell number as input to predict its cell heading.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""

import os
import numpy as np
from tensorflow import keras
import cv2
import time
from paths import DATA, checkPts, frames
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap
from DataGenerator2022 import DataGenerator2022
import random

### Uncomment next line to use CPU instead of GPU: ###
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class HeadingPredictorRGB(object):
    def __init__(self, checkPointFolder = None, loaded_checkpoint = None, imagesFolder = None, imagesParent = None, labelMapFile = None, data_name=None,
                 eval_ratio = 11.0 / 61.0, outputSize = 8, image_size=100, image_depth=3, dataSize = 0, seed=123456, batch_size = 20): #batch #epoch
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
        :param batch_size: Number of images in each batch fed into the model via Data Generator pipeline
        :param seed: Random seed to ensure that random splitting remains the same between training and validation
        """
        ### Set up paths and basic model hyperparameters
        self.checkpoint_dir = checkPointFolder + "2022HeadingPredict_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.outputSize = outputSize
        self.eval_ratio = eval_ratio
        self.learning_rate = 0.001
        self.image_size = image_size
        self.image_depth = image_depth
        self.seed = seed
        self.batch_size = batch_size
        self.num_eval = None
        # self.train_images = None
        # self.train_labels = None
        # self.eval_images = None
        # self.eval_labels = None
        self.data_name = data_name
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

    def prepDatasets(self):
        """Finds the cell labels associated with the files in the frames folder, and then sets up two
        data generators to preprocess data and produce the data in batches."""
        self.train_ds = DataGenerator2022(generateForCellPred = False, batch_size = self.batch_size)
        self.val_ds = DataGenerator2022(train=False, generateForCellPred = False, batch_size = self.batch_size)

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
    #         #batch_size= 1,
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
    #                 #batch_size=1,
    #                 write_images=False,
    #                 write_grads=True,
    #                 histogram_freq=1,
    #             ),
    #             keras.callbacks.TerminateOnNaN()
    #         ]
    #     )


    def train_withGenerator(self, training_generator, validation_generator, epoch = 20):
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
                                    write_images=False,
                                    write_grads=True
                                ),
                                keras.callbacks.TerminateOnNaN()
                ],
                            epochs= epoch)




    def cnn(self):
        """Builds a network that takes an image and produces the heading."""

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
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
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
            heading = self.labelMap.frameData[index]['heading']
            headingIndex = potentialHeadings.index(heading)
            if headingIndex is 8:  # the 0th index is 0 degree and is the same as the 8th index 360 degrees
                headingIndex = 0
            processed = self.cleanImage(image)
            pred, output = self.predictSingleImageAllData(processed)
            topThreePercs, topThreeHeadings = self.findTopX(3, output)
            topFivePercs, topFiveHeadings = self.findTopX(5, output)
            print("heading index =", headingIndex, "   pred =", pred)
            print("Top three:", topThreeHeadings, topThreePercs)
            print("Top five:", topFiveHeadings, topFivePercs)
            if pred == headingIndex:
                countPerfect += 1
            if headingIndex in topThreeHeadings:
                countTop3 += 1
            if headingIndex in topFiveHeadings:
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
    headingPredictor = HeadingPredictorRGB(
        data_name="FullData",
        checkPointFolder=checkPts,
        imagesFolder=frames,
        labelMapFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
        loaded_checkpoint="2022HeadingPredict_checkpoint-0615221151/TestingForVal-01-0.53.hdf5",
    )

    headingPredictor.buildNetwork()

    #for training:

    headingPredictor.prepDatasets()
    headingPredictor.train_withGenerator(headingPredictor.train_ds, headingPredictor.val_ds, epoch = 9)

    #for testing:

    #headingPredictor.test(1000)
