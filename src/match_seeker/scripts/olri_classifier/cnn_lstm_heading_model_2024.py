"""--------------------------------------------------------------------------------
cnn_lstm_heading_model_2024.py

Created: Summer 2024

This script can generate and train a heading prediction model with the 2024 style of dataset, using DataGeneratorLSTM.
It was built off of the cnn_heading_model_RGBinput.py from 2022, some of the methods still remain unchanged and
untested, so they might need some tweaking for them to work with this new model.
--------------------------------------------------------------------------------"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from src.match_seeker.scripts.olri_classifier.paths import *
from src.match_seeker.scripts.olri_classifier.DataGeneratorLSTM import DataGeneratorLSTM
from src.match_seeker.scripts.olri_classifier.imageFileUtils import makeFilename
from src.match_seeker.scripts.olri_classifier.frameCellMap import FrameCellMap

import time
import random
import csv

### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class HeadingPredictModelLSTM(object):
    def __init__(self, checkpoint_folder=None, loaded_checkpoint=None, images_folder=None, label_map_file=None, data_name=None,
                 output_size=8, image_size=224, seed=123456, batch_size=20):
        """
        :param checkpoint_folder: Destination path where checkpoints should be saved
        :param loaded_checkpoint: Name of the last saved checkpoint file inside checkPointFolder; used to continue training or conduct tests
        :param images_folder: Path of the folder containing all 95k image files
        :param label_map_file: Path of the txt file that contains the cell/heading/x y coordinate information for each of the 95k images
        :param data_name: Name that each checkpoint is saved under, indicates whether model is trained on all images or just a subset
        :param output_size: Number of output categories, 8 headings
        :param image_size: Target length of images after resizing, typically 100x100 pixels
        :param batch_size: Number of images in each batch fed into the model via Data Generator pipeline
        :param seed: Random seed to ensure that random splitting remains the same between training and validation
        """
        ### Set up paths
        self.checkpoint_dir = checkpoint_folder + "2024HeadingPredictLSTM_checkpoint-{}/".format(time.strftime("%m%d%y%H%M"))
        self.data_name = data_name
        self.frames = images_folder
        self.labelMapFile = label_map_file

        # Set up basic model hyperparameters
        self.outputSize = output_size
        self.learning_rate = 0.001
        self.image_size = image_size
        self.seed = seed
        self.batch_size = batch_size
        self.num_eval = None
        self.potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

        self.labelMap = None
        self.train_ds = None
        self.val_ds = None
        self.model = None

        if loaded_checkpoint:
            self.loaded_checkpoint = checkpoint_folder + loaded_checkpoint
        else:
            self.loaded_checkpoint = loaded_checkpoint

    def buildMap(self):
        """Builds dictionaries containing the corresponding cell, heading, and location information for each frame,
        saving it to self.labelMap. Untested in 2024"""
        self.labelMap = FrameCellMap(dataFile=self.labelMapFile)

    def prepDatasets(self):
        """Calls the Data Generator to create training and validation datasets for the model."""
        self.train_ds = DataGeneratorLSTM(framePath=framesDataPath, annotPath=textDataPath, seqLength=10,
                                          batch_size=self.batch_size, generateForCellPred=False)
        self.val_ds = DataGeneratorLSTM(framePath=framesDataPath, annotPath=textDataPath, seqLength=10,
                                        batch_size=self.batch_size, train=False, generateForCellPred=False)

    def buildNetwork(self):
        """Builds the network, saving it to self.model."""
        print (f"Tensorflow version: {tf.__version__}")
        print ("Calling buildNetwork", self.loaded_checkpoint)
        if self.loaded_checkpoint:
            self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
            print ("Got past the model loading")
            self.model.summary()
        else:
            self.model = self.CNN_LSTM()

        self.model.compile(
            loss= keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"])

    def train(self, epochs = 20):
        """Begins training of the model. Defaults to 20 epochs"""
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=self.val_ds,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.keras",  # Name for checkpoint
                    save_freq="epoch"  # save every epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    write_images=False,
                    # write_grads=True,
                    histogram_freq=1,
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )

    def train_withGenerator(self, training_generator, validation_generator, epoch = 20):
        """Not tested in 2024. Consider changing or removing."""
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
                            epochs=epoch)

    def CNN_LSTM(self):
        """Builds the CNN + LSTM model."""

        cnnLSTM = keras.models.Sequential()

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
          filters=16,  # Reduced from 64
          kernel_size=(3, 3),
          strides=(1, 1),
          activation="relu",
          padding="same",
          data_format="channels_last",
        ), input_shape=[self.batch_size, self.image_size, self.image_size, 3]))

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
          pool_size=(2, 2),
          strides=(2, 2),
          padding="same",
        )))

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))  # Reduced dropout rate

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Conv2D(
          filters=8,  # Reduced from 32
          kernel_size=(3, 3),
          strides=(1, 1),
          activation="relu",
          padding="same",
          data_format="channels_last"
        )))

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D(
          pool_size=(2, 2),
          strides=(2, 2),
          padding="same",
        )))

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))  # Reduced dropout rate

        cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        cnnLSTM.add(keras.layers.LSTM(5, activation="relu"))  # Adjust LSTM units as needed

        cnnLSTM.add(keras.layers.Dense(units=self.outputSize, activation='sigmoid'))
        cnnLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        cnnLSTM.summary()

        self.model = cnnLSTM

        # ---YET TO TRY--- (new cnn lstm method based on bleed ai example)
        # cnnLSTM.add(keras.layers.ConvLSTM2D(
        #     filters=128,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     activation="relu",
        #     padding="same",
        #     data_format="channels_last",
        #
        # ), input_shape=[self.batch_size, self.image_size, self.image_size, 1])
        # cnnLSTM.add(keras.layers.MaxPooling3D(
        #     pool_size=(2, 2),
        #     strides=(2, 2),
        #     padding="same"
        # ))
        # cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
        #
        # cnnLSTM.add(keras.layers.ConvLSTM2D(
        #     filters=64,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     activation="relu",
        #     padding="same",
        #     data_format="channels_last",
        #
        # ), input_shape=[self.batch_size, self.image_size, self.image_size, 1])
        # cnnLSTM.add(keras.layers.MaxPooling3D(
        #     pool_size=(2, 2),
        #     strides=(2, 2),
        #     padding="same"
        # ))
        # cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
        #
        # cnnLSTM.add(keras.layers.ConvLSTM2D(
        #     filters=32,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     activation="relu",
        #     padding="same",
        #     data_format="channels_last",
        #
        # ), input_shape=[self.batch_size, self.image_size, self.image_size, 1])
        # cnnLSTM.add(keras.layers.MaxPooling3D(
        #     pool_size=(2, 2),
        #     strides=(2, 2),
        #     padding="same"
        # ))
        # cnnLSTM.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.4)))
        #
        # cnnLSTM.add(keras.layers.Flatten())
        # cnnLSTM.add(keras.layers.Dense(units=self.outputSize, activation='sigmoid'))
        # cnnLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # cnnLSTM.summary()

        return cnnLSTM

    def predictSingleImageBatchAllData(self, images):
        """Given a batch of images, converts it to be suitable for the network, then runs the model and returns
        the resulting prediction as tuples of index of prediction and list of predictions."""
        cleanImages = []
        for image in images:
            cleanImage = self.cleanImage(image, 224)
            cleanImages.append(cleanImage)
        listed = np.asarray([cleanImages])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
        return maxIndex, modelPredict[0]

    def cleanImage(self, image, image_size=224):
        """Process a single image into the correct input form for 2020 model, mainly used for testing."""
        shrunkenIm = cv2.resize(image, (image_size, image_size))
        recoloredIm = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2RGB)
        processedIm = recoloredIm / 255.0
        return processedIm

    # TODO: The methods below were not tested in 2024, they were just kept from the 2022 file. Consider editing or removing

    def findTopX(self, x, num_list):
        """Given a number and a list of numbers, this finds the x largest values in the number list, and reports
        both the values, and their positions in the numList."""
        topVals = [0.0] * x
        topIndex = [None] * x
        for i in range(len(num_list)):
            val = num_list[i]

            for j in range(x):
                if topIndex[j] is None or val > topVals[j]:
                    break
            if val > topVals[x - 1]:
                topIndex.insert(j, i)
                topVals.insert(j, val)
                topIndex.pop(-1)
                topVals.pop(-1)
        return topVals, topIndex

    def findBottomX(self, x, num_list):
        """Given a number and a list of numbers, this finds the x smallest values in the number list, and reports
        both the values, and their positions in the numList."""
        topVals = [1e8] * x
        topIndex = [None] * x
        for i in range(len(num_list)):
            val = num_list[i]

            for j in range(x):
                if topIndex[j] is None or val < topVals[j]:
                    break
            if val < topVals[x - 1]:
                topIndex.insert(j, i)
                topVals.insert(j, val)
                topIndex.pop(-1)
                topVals.pop(-1)
        return topVals, topIndex

    def test(self, n, random_choose=True):
        """This runs each of the random n images in the folder of frames through the cell-output network, reporting how
        often the correct cell was produced0, and how often the correct heading was in the top 3 and top 5.
        Or when setting randomChoose to be false, it tests the model on the n-th image."""

        self.buildMap()
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        countPerfect = 0
        countTop3 = 0
        countTop5 = 0
        if random_choose:
            iterations = n
        else: iterations = 1
        for i in range(iterations):
            if random_choose:
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
            pred, output = self.predictSingleImageBatchAllData(image)
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

    def testnImagesAllHeadings(self, n):
        """
        Tests the model on n randomly selected photos per heading. Calculates and plots
        the success rate per heading and then displays the images used to test the worst
        performing heading
        :param n: number of randomly selected photos to test per heading
        :return:
        """
        self.buildMap()
        n_frames_map = self.labelMap.selectNFramesAllHeadings(n)

        successMap = {}
        failedMap = {}
        frameProbability = {}
        frameTop3PredProb = {}
        for heading in n_frames_map:
            headingFrames = n_frames_map.get(heading)
            perfMapHeading, failedMapHeading, frameProbabilityHeading, frameTop3PredProbHeading = self.testOneHeading(heading, headingFrames)
            successMap.update(perfMapHeading)
            failedMap.update(failedMapHeading)
            frameProbability.update(frameProbabilityHeading)
            frameTop3PredProb.update(frameTop3PredProbHeading)
        headings, successRates = self.getAllSuccessRates(successMap, failedMap)
        self.plotSuccessRates(headings, successRates)
        self.showImagesNWorstHeadings(headings, successRates, n_frames_map, successMap, failedMap, frameProbability, bottom_n= 3)
        self.logNWorstHeadings("HeadingPredBottom3AllFrames", headings, successRates, n_frames_map, successMap, failedMap, frameProbability, frameTop3PredProb, bottomN = 3)

    def testnImagesOneCell(self, heading, n):
        """
        Tests n images belonging to one heading. Displays the n images of the heading used in testing and logs
        performance metrics (frame number, success rate, predicted heading, probability of actual, probability of predicted,
        top 3 predicted headings, top 3 probabilities) in a CSV file located in /res/csvLogs/headingPredictorRGBTestLogs
        :param heading: heading number of cell to be tested alone
        :param n: number of images of the specific heading used to test the model
        """
        self.buildMap()
        headingFrames = self.labelMap.selectNFramesOneHeading(heading, n)
        successMap, failedMap, frameProbability, frameTop3PredProb = self.testOneHeading(heading, headingFrames)

        totalPred, successRate = self.getHeadingSuccessRate(heading, successMap, failedMap)

        # get list of nested lists containing [list of failed frames], [list of failed predictions each frame]
        listOfFramesFailedPred = failedMap.get(heading)

        # create new map with failed frame number as keys, failed prediction per frame as values
        headingFailFramesMap = {list[0]: list[1] for list in listOfFramesFailedPred}
        successFrames = successMap.get(heading)

        self.showImagesOneHeading(heading, headingFrames, frameProbability, successFrames, headingFailFramesMap)

        logPath = logs + "headingPredictorRGBTestLogs/"
        csvLog = open(logPath + "testCell" + str(heading) + "-{}.csv".format(time.strftime("%m%d%y%H%M")), 'w')
        filewriter = csv.writer(csvLog)
        filewriter.writerow(
            ["Frame", "Actual Heading", "Predicted Heading", "Success", "Heading Success Rate", "Prob Actual", "Prob Predicted",
             "Top 3 Pred", "Top 3 Prob"])
        self.logOneHeading(filewriter, heading, str(successRate), headingFrames, successFrames, headingFailFramesMap, frameProbability, frameTop3PredProb)

    def convertIndexListtoHeading(self, list):
        """
        Helper function to convert a list of indices of headings into heading numbers
        :param list: list of heading indices
        :return headings: list of heading numbers
        """
        headings = []
        for i in list:
            i = int(i)
            headings.append(self.potentialHeadings[i])
        return headings

    def convertHeadingListtoIndex(self, list):
        """
        Helper function to convert a list of heading numbers into heading indices
        :param list: list of heading numbers
        :return headings: list of heading indices
        """
        headingIndices = []
        for i in list:
            i = int(i)
            headingIndices.append(self.potentialHeadings.index(i))
        return headingIndices

    def testOneHeading(self, heading, frames_list):
        """ Tests the performance of the RGB heading model in predicting one given heading.
        Takes in a heading number and a list of frames corresponding to the heading, and returns
        dictionaries recording successful and unsuccessful frames, prediction probabilities per frame,
        and top three predictions/probabilities per frame
        :param heading: heading number (int)
        :param frames_list: List of frames corresponding to the heading to be tested
        :return:
            successMap: Map of perfectly predicted test cases, heading number as keys and list of perfectly predicted frames as values
            failedMap: Map of incorrectly predicted test cases, heading number as keys and list of lists containing incorrect predictions and frames as values
            frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each heading generated by the model as values
            frameTop3PredProd: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Headings]) as values
        """
        successMap = {}
        failedMap = {}
        frameProbability = {}
        frameTop3PredProb = {}
        for frame in frames_list:
            imFile = makeFilename(self.frames, frame)
            print(imFile)
            image = cv2.imread(imFile)
            if image is None:
                print(" image not found")
                continue
            pred, output = self.predictSingleImageBatchAllData(image)
            pred = self.potentialHeadings[int(pred)]
            topThreePercs, topThreeHeadings = self.findTopX(3, output)
            topThreeHeadings = self.convertIndexListtoHeading(topThreeHeadings)
            frameProbability[frame] = output
            frameTop3PredProb[frame] = [topThreePercs, topThreeHeadings]
            if pred == heading:
                prevSuccessList = successMap.get(heading, [])
                prevSuccessList.append(frame)
                successMap[heading] = prevSuccessList
            else:
                prevFails = failedMap.get(heading, [])
                prevFails.append([frame, pred])
                failedMap[heading] = prevFails
        return successMap, failedMap, frameProbability, frameTop3PredProb

    def getHeadingSuccessRate(self, heading, success_map, failed_map):
        """
        Calculates the success rate of one heading. Returns the number of total predictions for that heading and the success rate
        using data from the success map and failed predictions map. If the heading is missing from the dataset (i.e. no predictions
        then it returns 0 and 0.0 for total predictions and success rate.
        :param heading: actual heading number
        :param success_map: Map of perfectly predicted test cases, heading number as keys and list of perfectly predicted frames as values
        :param failed_map: Map of incorrectly predicted test cases, heading number as keys and list of lists containing incorrect predictions and frames as values
        """
        totalPred = 0
        if heading in failed_map:
            totalPred += len(failed_map[heading])
        if heading in success_map:
            totalPred += len(success_map[heading])
        if totalPred > 0:
            successRate = len(success_map.get(heading, 0)) / float(totalPred)
            return totalPred, successRate
        return totalPred, 0.0

    def getAllSuccessRates(self, success_map, failed_map, exclude_missing=True, exclude_perfect=False):
        """
        Calculates the success rate per cell number category, with the option to
        exclude cells that have a 1.0 perfect success rate or cells that do not show up in the randomly
        generated test cases.

        :param success_map: Map of perfectly predicted test cases, cell number as keys and list of perfectly predicted frames as values
        :param failed_map: Map of incorrectly predicted test cases, cell number as keys and list of lists containing incorrect predictions and frames as values
        :param exclude_missing: boolean value deciding whether to include cells not tested
        :param exclude_perfect: boolean value deciding whether to include cells with 100% accuracy of prediction
        :return: List of cells (str) and list of success rates (float) of the cell in the same index
        """
        successRates = []
        headings = []
        for head in self.potentialHeadings:
            totalPred, successRate = self.getHeadingSuccessRate(head, success_map, failed_map)
            if totalPred > 0:
                successRate = len(success_map.get(head, 0)) / float(totalPred)
                if successRate == 1.0:
                    if exclude_perfect:
                        continue
                headings.append(str(head))
                successRates.append(successRate)
            else:
                if not exclude_missing:
                    successRates.append(-1)
                    headings.append(str(head))
                continue
        print("Success rates: ", successRates)
        print("Number of headings: ", len(headings), "Number of success rates: ", len(successRates))
        return headings, successRates

    def plotSuccessRates(self, headings, success_rates):
        """
        Plots success rates against cell number using matplotlib.
        :param headings: List of cells, each entry MUST be a string and indices must be aligned with successRates
        :param success_rates: List of success rates per cell, indices must be aligned with cells
        :return:
        """
        plt.scatter(headings, success_rates)
        plt.xlabel('Heading Number', fontsize=16)
        plt.ylabel('Success Rate %', fontsize=16)
        plt.title('Success Rate Per Heading', fontsize=16)
        plt.show()

    def showImagesNWorstHeadings(self, headings, success_rates, frames_map, success_map, failed_map, frame_probability, bottom_n=3):
        """
        Displays the n photos used in testnImagesEachHeading of the bottomN number of headings with the lowest accuracies.
        Each image window displays whether the frame was successfully or unsuccessfully predicted, the
        predicted heading (if unsuccessful), the probability of the model predicting the actual heading from the image,
        and the probability of the model predicting the wrong cell (if the photo was unsuccessfully predicted).
        Method is meant to be called inside testnImagesAllHeadings
        :param headings: List of headings tested, indices align with successRates
        :param success_rates: List of success rates per heading, indices align with headings
        :param frames_map: Map of n randomly selected frames per heading with headings for keys, list of frames as values
        :param success_map: Map of successful frames predicted per heading with headings for keys, list of frames as values
        :param failed_map: Map of unsuccessfully predicted frames per heading, with headings for keys, list of lists containing
        failed predicted heading and frame number
        :param frame_probability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities
        for each heading generated by the model as values
        :param bottom_n: number of n lowest performing headings
        :return:
        """
        bottomNRate, bottomNCellID = self.findBottomX(bottom_n, success_rates)
        print(bottomNRate)
        print(bottomNCellID)
        for i in range(bottom_n):
            worstSuccessRate = bottomNRate[i]
            indexOfWorstHeading = bottomNCellID[i]
            worstHeading = int(headings[indexOfWorstHeading])

            #get list of nested lists containing [list of failed frames], [list of failed predictions each frame]
            listOfFramesFailedPred = failed_map.get(worstHeading, [])

            #create new map with failed frame number as keys, failed prediction per frame as values
            headingFailFramesMap ={list[0]:list[1] for list in listOfFramesFailedPred}

            print('Worst Performing Heading: ', worstHeading, " Success Rate: ", worstSuccessRate)
            framesList = frames_map.get(worstHeading)
            successFrames = success_map.get(worstHeading)

            self.showImagesOneHeading(worstHeading, framesList, frame_probability, successFrames, headingFailFramesMap)

    def showImagesOneHeading(self, heading, framesList, frameProbability, successFrames, headingFailFramesMap):
        """
        Displays the frames corrensponding to one heading used in testing the RGB cell predictor. Each window
        shows information about each frame's prediction result such as predicted heading, prob of prediction,
        prof of actual (if failed prediction), and failed prediction
        :param heading: heading whose corresponding photos will be shown
        :param framesList: list of frames corresponding to the given heading, each frame will be displayed individually
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each heading generated by the model as values
        :param successFrames: List of successfully predicted frames corresponding to the heading
        :param headingFailFramesMap: Map of unsuccessfully predicted frames, with frame numbers for keys and the failed  heading prediction for the frame as values
        :return:
        """
        for frame in framesList:
            imFile = makeFilename(self.frames, frame)
            image = cv2.imread(imFile)
            if image is None:
                print(" image not found")
                continue
            headingInx = self.potentialHeadings.index(heading)
            probActualHeading = frameProbability[frame][headingInx]
            if frame in successFrames:
                cv2.imshow("Frame: " + str(frame) + " Heading " + str(heading) + " Success, Prob Actual: " + str(
                    probActualHeading), image)
            else:
                predHeading = headingFailFramesMap.get(frame)
                predHeadingInd = self.potentialHeadings.index(predHeading)
                probForWrongPrediction = frameProbability[frame][predHeadingInd]
                cv2.imshow("FR: " + str(frame) + " Fail " + str(heading) + " Pred " + str(predHeading) + ", Prob: " + str(
                    probForWrongPrediction) + ", Prob Actual: " + str(probActualHeading), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def logOneHeading(self, filewriter, heading, headingSuccessRate, framesList, successFrames, headingFailFramesMap,
                      frameProbability, frameTop3PredProb):
        """
        Takes in a csv filewriter object and logs test performance information of one cell into the csv associated with filewriter.
        Records the frame number, actual cell, predicted cell, cell success rate, probability of prediction, top 3 predictions, and top 3 probabilties
        for the specified cell that was tested. Meant to be called in other test functions for ease of logging. CSV must be closed outside this function
        for logs to be successfully saved.

        :param filewriter: CSV filewriter object associated with the csv log
        :param heading: heading number of the tested heading
        :param headingSuccessRate: success rate of the tested heading
        :param framesList: list of frame names used in testing that all correspond to heading
        :param successFrames: list of successfully predicted frames belonging to the heading
        :param headingFailFramesMap Map of unsuccessfully predicted frames, with frame numbers for keys and the failed heading prediction for the frame as values
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each heading generated by the model as values
        :param frameTop3PredProb: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Headings]) as values
        :return:
        """
        for frame in framesList:
            headingInx = self.potentialHeadings.index(heading)
            probActualHeading = frameProbability[frame][headingInx]
            frameTop3Prob = frameTop3PredProb[frame][0]
            frameTop3PredHeading = frameTop3PredProb[frame][1]
            if frame in successFrames:
                filewriter.writerow(
                    [frame, str(heading), str(heading), "T", str(headingSuccessRate), str(probActualHeading),
                     str(probActualHeading), str(frameTop3PredHeading), str(frameTop3Prob)])
            else:
                predHeading = headingFailFramesMap.get(frame)
                predHeadingInd = self.potentialHeadings.index(predHeading)
                probForWrongPrediction = frameProbability[frame][predHeadingInd]
                filewriter.writerow(
                    [frame, str(heading), str(predHeading), "F", str(headingSuccessRate), str(probActualHeading),
                     str(probForWrongPrediction), str(frameTop3PredHeading), str(frameTop3Prob)])

    def logNWorstHeadings(self, filename, headings, successRates, framesMap, successMap, failedMap, frameProbability,
                          frameTop3PredProb, bottomN=1):
        """
        Creates a CSV Log to record testing performance metrics (frame number, predicted heading, actual heading, heading success rate, probability of actual,
        probability of predicted, top three predicted headings, top three probabilities) inside res/csvLogs/headingPredictorRGBTestLogs.
        Meant to be called inside testNImagesAllHeadings for logging purposes.

        :param filename: String filename name that logs will be saved under inside res/csvLogs/headingPredictorRGBTestLogs
        :param headings: List of headings tested, indices align with successRates
        :param successRates: List of success rates per heading, indices align with cells
        :param framesMap: Map of n randomly selected frames per heading with headings for keys, list of frames as values
        :param successMap: Map of successful frames predicted per heading with headings for keys, list of frames as values
        :param failedMap: Map of unsuccessfully predicted frames per heading, with headings for keys, list of lists containing
        failed predicted heading and frame number
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities
        for each heading generated by the model as values
        :param frameTop3PredProb: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Headings]) as values
        :param bottomN: number of n lowest performing headings
        """
        dirTimeStamp = "{}".format(time.strftime("%m%d%y%H%M"))
        logPath = logs + "headingPredictorRGBTestLogs/"
        csvLog = open(logPath + filename + "-" + dirTimeStamp + ".csv", "w")
        filewriter = csv.writer(csvLog)
        filewriter.writerow(
            ["Frame", "Actual Heading", "Predicted Heading", "Success", "Heading Success Rate", "Prob Actual", "Prob Predicted",
             "Top 3 Pred", "Top 3 Prob"])

        bottomNRate, bottomNCellID = self.findBottomX(bottomN, successRates)
        for i in range(bottomN):
            worstSuccessRate = bottomNRate[i]
            indexOfWorstHeading = bottomNCellID[i]
            worstHeading = int(headings[indexOfWorstHeading])

            listOfFramesFailedPred = failedMap.get(worstHeading)
            HeadingFailFramesMap = {list[0]: list[1] for list in listOfFramesFailedPred}

            framesList = framesMap.get(worstHeading, [])
            successFrames = successMap.get(worstHeading)

            self.logOneHeading(filewriter, worstHeading, worstSuccessRate, framesList, successFrames, HeadingFailFramesMap,
                            frameProbability, frameTop3PredProb)
        csvLog.close()

    def evaluateModel(self):
        print("Evaluating with training data")
        results_val = self.model.evaluate(self.train_ds)
        print(results_val)
        print("Evaluating with validation data")
        results_val = self.model.evaluate(self.val_ds)
        print(results_val)


# def loading_bar(start,end, size = 20):
#     # Useful when running a method that takes a long time
#     loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
#     if start % 10 == 0:
#         print(loadstr)

if __name__ == "__main__":
    headingPredictor = HeadingPredictModelLSTM(
        data_name="HeadingPredict224",
        checkpoint_folder=checkPts,
        images_folder=framesDataPath,
        label_map_file=DATA + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
        loaded_checkpoint="2024HeadingPredict_checkpoint-0717241135/TestHeadingInCellPredAdam224Corrected-61-0.07.keras",
    )

    headingPredictor.buildNetwork()

    # For training:
    # Call prepDatasets
    # Call a train method

    # For evaluation
    headingPredictor.prepDatasets()
    headingPredictor.evaluateModel()

    # For testing:
    # The testing methods have not been changed from the 2022 file
