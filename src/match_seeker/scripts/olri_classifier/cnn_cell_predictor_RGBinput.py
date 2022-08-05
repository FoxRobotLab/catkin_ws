"""--------------------------------------------------------------------------------
cnn_cell_predictor_RGBinput.py

Updated: Summer 2022

This file can build and train CNN and load checkpoints/models for predicting cells.
The model was originally created in 2019 that takes a picture and its heading as input to predict its cell number.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""

import cv2
import os
import numpy as np
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from paths import DATA, checkPts, frames
from imageFileUtils import makeFilename, extractNum
from frameCellMap import FrameCellMap
from DataGenerator2022 import DataGenerator2022
from DataBalancing2022 import DataBalancer
import random
import csv
import tensorflow as tf

### Uncomment next line to use CPU instead of GPU: ###
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CellPredictorRGB(object):
    def __init__(self, checkPointFolder = None, loaded_checkpoint = None, imagesFolder = None, imagesParent = None, labelMapFile = None, data_name=None,
                 eval_ratio=11.0 / 61.0, outputSize=271, image_size=224, image_depth=4, dataSize = 0, batch_size = 10, seed=4359):
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
        self.data_name = data_name
        self.dataSize = dataSize #len of ...?
        self.frames = imagesFolder
        self.framesParent = imagesParent
        self.labelMapFile = labelMapFile
        self.labelMap = None
        self.train_ds = None
        self.val_ds = None
        if loaded_checkpoint is not None:
            self.loaded_checkpoint = checkPointFolder + loaded_checkpoint
        else: self.loaded_checkpoint = loaded_checkpoint

    def buildMap(self):
        """Builds dictionaries containing the corresponding cell, heading, and location information for each frame,
        saving it to self.labelMap."""
        self.labelMap = FrameCellMap(dataFile=self.labelMapFile)

    #not compatible with tensorflow 1, also seems to do different preprocessing from the next prepDatasets function
    # def prepDatasets(self):
    #     """Finds the cell labels associated with the files in the frames folder, and then sets up two
    #     data generators to produce the data in batches."""
    #     self.buildMap()
    #     #From Tensorflow website:
    #     #Labels should be sorted according to the alphanumeric order of the image file paths (obtained via os.walk(directory) in Python)
    #     #given a directory, os.walk returns list of dirpaths, list of dirnames and list of filenames as tuples, the generated lists are in the same order in every run
    #     print(self.frames, self.framesParent)
    #     imageFiles = os.listdir(self.frames)
    #     imageFiles.sort()
    #     cellLabels = [self.labelMap.frameData[fNum]['cell'] for fNum in map(extractNum, imageFiles)]
    #
    #     self.train_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels, subset="training",
    #                                                              label_mode = 'int',
    #                                                              validation_split=0.2,  seed=self.seed,
    #                                                              image_size=(self.image_size, self.image_size),
    #                                                              batch_size=self.batch_size)
    #     self.train_ds = self.train_ds.map(lambda x, y: (x /255., y)) #we shall use keras rescale layer instead of this line
    #
    #     #displays images and labels in first batch
    #
    #     # for images, labels in self.train_ds.take(1):
    #     #     for i in range(self.batch_size):
    #     #         print("Label: ", labels[i])
    #     #         print("Image: ", images[i].numpy())
    #     #         image = cv2.convertScaleAbs(images[i].numpy()) # cv2 cannot show floating points
    #     #         cv2.imshow("Label"+str(labels[i]) , image)
    #     #         cv2.waitKey(0)
    #
    #     self.val_ds = keras.utils.image_dataset_from_directory(self.framesParent, labels=cellLabels, subset="validation",
    #                                                            label_mode='int',
    #                                                            validation_split=0.2, seed=self.seed,
    #                                                            image_size=(self.image_size, self.image_size),
    #                                                            batch_size=self.batch_size)
    #     self.val_ds = self.val_ds.map(lambda x, y: (x / 255., y)) #use keras rescale layer instead
    #
    #     #methods that does not work for listing filenames of images of dataset
    #
    #     # iterator_helper = self.val_ds.make_one_shot_iterator()
    #     # with tf.Session() as sess:
    #     #     filename_temp = iterator_helper.get_next()
    #     #     print(filename_temp)
    #     #     print(sess.run[filename_temp])
    #
    #     # for i, element in enumerate(self.val_ds.as_numpy_iterator()):
    #     #     print(element)
    #     #     if i>50:
    #     #         break
    #
    #     # file_paths = self.val_ds.file_paths
    #     # print(file_paths[0:10])
    #
    #     # displays images and labels in first batch
    #     # for images, labels in self.val_ds.take(1):
    #     #     for i in range(self.batch_size):
    #     #         print("Label: ", labels[i])
    #     #         print("Image: ", images[i].numpy())
    #     #         image = cv2.convertScaleAbs(images[i].numpy()) # cv2 cannot show floating points
    #     #         cv2.imshow("Label"+str(labels[i]) , image)
    #     #         cv2.waitKey(0)


    def prepDatasets(self):
        """Finds the cell labels associated with the files in the frames folder, and then sets up two
        data generators to preprocess data and produce the data in batches."""
        self.train_ds = DataGenerator2022(batch_size = self.batch_size, cellPredWithHeadingIn = True)
        self.val_ds = DataGenerator2022(batch_size = self.batch_size, train = False, cellPredWithHeadingIn = True)

    def buildNetwork(self):
        """Builds the network, saving it to self.model."""
        if self.loaded_checkpoint is not None:
            self.model = keras.models.load_model(self.loaded_checkpoint, compile=False)
            #print("---Loading weights---")
            self.model.load_weights(self.loaded_checkpoint)
        else:
            self.model = self.cnn()  # CNN

        self.model.compile(
            loss= keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            metrics=["accuracy"])

    def train(self, epochs = 20):
        """Sets up the loss function and optimizer, and then trains the model on the current training data. Quits if no
        training data is set up yet."""
        balancer = DataBalancer()
        weights = balancer.getClassWeightCells()
        self.model.fit(
            self.train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=self.val_ds,
            class_weight = weights,
            callbacks=[
                keras.callbacks.History(),
                keras.callbacks.ModelCheckpoint(
                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                    save_freq="epoch"  # save every epoch
                ),
                keras.callbacks.TensorBoard(
                    log_dir=self.checkpoint_dir,
                    write_images=False,
                    write_grads=True
                ),
                keras.callbacks.TerminateOnNaN()
            ]
        )


    def train_withGenerator(self, epochs = 20 ):
        balancer = DataBalancer()
        weights = balancer.getClassWeightCells()
        self.model.fit_generator(generator=self.train_ds,
                            validation_data=self.val_ds,
                            use_multiprocessing=True,
                            workers=6,
                            class_weight=weights,
                            callbacks=[
                                keras.callbacks.History(),
                                keras.callbacks.ModelCheckpoint(
                                    self.checkpoint_dir + self.data_name + "-{epoch:02d}-{val_loss:.2f}.hdf5",
                                    # save_freq="epoch"  # save every n epoch
                                ),
                                keras.callbacks.TensorBoard(
                                    log_dir=self.checkpoint_dir,
                                    write_images=False,
                                    write_grads=True
                                ),
                                keras.callbacks.TerminateOnNaN()
                ],
                            epochs= epochs)




    def cnn(self):
        """Builds a network that takes an image and produces the cell number."""

        model = keras.models.Sequential()

        #rescale and resize layers not compatible with tensorflow 1, so they're done in prepDatasets function.
        # model.add(keras.layers.Resizing(
        #         self.image_size, self.image_size, interpolation="bilinear", crop_to_aspect_ratio=False, **kwargs))
        # model.add(keras.layers.Rescaling(scale = 1./255, offset=0.0,
        #                                  input_shape=[self.image_size, self.image_size, self.image_depth]))

        model.add(keras.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            activation="relu",
            padding="same",
            data_format="channels_last"
            ,input_shape=[self.image_size, self.image_size, self.image_depth]
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
        # print(model.input.shape)
        return model


    def predictSingleImageAllData(self, image):
        """Given an image, converts it to be suitable for the network, then runs the model and returns
        the resulting prediction as tuples of index of prediction and list of predictions."""
        cleanimage = self.cleanImage(image)
        listed = np.array([cleanimage])
        modelPredict = self.model.predict(listed)
        maxIndex = np.argmax(modelPredict)
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

    def findBottomX(self, x, numList):
        """Given a number and a list of numbers, this finds the x smallest values in the number list, and reports
        both the values, and their positions in the numList."""
        topVals = [1e8] * x
        topIndex = [None] * x
        for i in range(len(numList)):
            val = numList[i]

            for j in range(x):
                if topIndex[j] is None or val < topVals[j]:
                    break
            if val < topVals[x - 1]:
                topIndex.insert(j, i)
                topVals.insert(j, val)
                topIndex.pop(-1)
                topVals.pop(-1)
        return topVals, topIndex

    def test(self, n, randomChoose = True):
        """
        This runs each of the random n images in the folder of frames through the cell-output network, reporting how
        often the correct cell was prodplt.show()uced, and how often the correct heading was in the top 3 and top 5.
        Or when setting randomChoose to be false, it tests the model on the n-th image
        :param n: random n images or the n-th image if randomChoose is set to be False
        :param randomChoose: boolean value deciding whether we are testing on n images or the n-th image on the model
        :return:
        """

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
            pred, output = self.predictSingleImageAllData(image)
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


    def testnImagesAllCells(self, n):
        """
        Tests the model on n randomly selected photos per cell. Calculates and plots
        the success rate per cell and then displays the images used to test the worst
        performing cell
        :param n: number of randomly selected photos to test per cell
        :return:
        """
        self.buildMap()
        n_frames_map = self.labelMap.selectNFramesAllCells(n)

        successMap = {}
        failedMap = {}
        frameProbability = {}
        frameTop3PredProb = {}
        for cell in n_frames_map:
            cell_frames = n_frames_map.get(cell)
            perfMapCell, failedMapCell, frameProbabilityCell, frameTop3PredProbCell = self.testOneCell(cell, cell_frames)
            successMap.update(perfMapCell)
            failedMap.update(failedMapCell)
            frameProbability.update(frameProbabilityCell)
            frameTop3PredProb.update(frameTop3PredProbCell)
        cells, successRates = self.getAllSuccessRates(successMap, failedMap)
        self.plotSuccessRates(cells, successRates)
        self.showImagesNWorstCells(cells, successRates, n_frames_map, successMap, failedMap, frameProbability, bottomN = 5)
        self.logNWorstCells("CellPredBottom5AllFrames", cells, successRates, n_frames_map, successMap, failedMap, frameProbability, frameTop3PredProb, bottomN = 5)


    def testnImagesOneCell(self, cell, n):
        """
        Tests n images belonging to one cell. Displays the n images of the cell used in testing and logs
        performance metrics (frame number, success rate, predicted cell, probability of actual, probability of predicted,
        top 3 predicted cells, top 3 probabilities) in a CSV file located in /res/csvLogs/cellPredictorRGBTestLogs
        :param cell: cell number of cell to be tested alone
        :param n: number of images of the specific cell used to test the model
        """
        self.buildMap()
        cellFrames = self.labelMap.selectNFramesOneCell(cell, n)
        successMap, failedMap, frameProbability, frameTop3PredProb = self.testOneCell(cell, cellFrames)

        totalPred, successRate = self.getCellSuccessRate(cell, successMap, failedMap)

        # get list of nested lists containing [list of failed frames], [list of failed predictions each frame]
        listOfFramesFailedPred = failedMap.get(cell)

        # create new map with failed frame number as keys, failed prediction per frame as values
        cellFailFramesMap = {list[0]: list[1] for list in listOfFramesFailedPred}
        successFrames = successMap.get(cell)

        self.showImagesOneCell(cell, cellFrames, frameProbability, successFrames, cellFailFramesMap)

        logPath = "../../res/csvLogs2022/cellPredictorRGBTestLogs/"
        csvLog = open(logPath + "testCell" + str(cell) + "-{}.csv".format(time.strftime("%m%d%y%H%M")), 'w')
        filewriter = csv.writer(csvLog)
        filewriter.writerow(
            ["Frame", "Actual Cell", "Predicted Cell", "Success", "Cell Success Rate", "Prob Actual", "Prob Predicted",
             "Top 3 Pred", "Top 3 Prob"])
        self.logOneCell(filewriter, cell, str(successRate), cellFrames, successFrames, cellFailFramesMap, frameProbability, frameTop3PredProb)



    def testOneCell(self, cell, framesList):
        """ Tests the performance of the RGB cell model in predicting one given cell.
        Takes in a cell number and a list of frames corresponding to the cell, and returns
        dictionaries recording successful and unsuccessful frames, prediction probabilities per frame,
        and top three predictions/probabilities per frame
        :param cell: Cell number (int)
        :param framesList: List of frames corresponding to the cell to be tested
        :return:
            successMap: Map of perfectly predicted test cases, cell number as keys and list of perfectly predicted frames as values
            failedMap: Map of incorrectly predicted test cases, cell number as keys and list of lists containing incorrect predictions and frames as values
            frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each cell generated by the model as values
            frameTop3PredProd: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Cells]) as values
        """
        successMap = {}
        failedMap = {}
        frameProbability = {}
        frameTop3PredProb = {}
        for frame in framesList:
            imFile = makeFilename(self.frames, frame)
            print(imFile)
            image = cv2.imread(imFile)
            if image is None:
                print(" image not found")
                continue
            pred, output = self.predictSingleImageAllData(image)
            topThreePercs, topThreeCells = self.findTopX(3, output)
            frameProbability[frame] = output
            frameTop3PredProb[frame] = [topThreePercs, topThreeCells]
            if pred == cell:
                prevSuccessList = successMap.get(cell, [])
                prevSuccessList.append(frame)
                successMap[cell] = prevSuccessList
            else:
                prevFails = failedMap.get(cell, [])
                prevFails.append([frame, pred])
                failedMap[cell] = prevFails
        return successMap, failedMap, frameProbability, frameTop3PredProb


    def cleanImage(self, image, imageSize=224):
        """Process a single image into the correct input form for 2020 model, mainly used for testing."""
        shrunkenIm = cv2.resize(image, (imageSize, imageSize))
        processedIm = shrunkenIm / 255.0
        return processedIm

    def getCellSuccessRate(self, cell, successMap, failedMap):
        """
        Calculates the success rate of one cell. Returns the number of total predictions for that cell and the success rate
        using data from the success map and failed predictions map. If the cell is missing from the dataset (i.e. no predictions
        then it returns 0 and 0.0 for totalPred and success rate.
        :param cell: actual cell number
        :param successMap: Map of perfectly predicted test cases, cell number as keys and list of perfectly predicted frames as values
        :param failedMap: Map of incorrectly predicted test cases, cell number as keys and list of lists containing incorrect predictions and frames as values
        """
        totalPred = 0
        if cell in failedMap:
            totalPred += len(failedMap[cell])
        if cell in successMap:
            totalPred += len(successMap[cell])
        if totalPred > 0:
            successRate = len(successMap.get(cell, 0)) / float(totalPred)
            return totalPred, successRate
        return totalPred, 0.0

    def getAllSuccessRates(self, successMap, failedMap, excludeMissing = True, excludePerfect = False):
        """
        Calculates the success rate per cell number category, with the option to
        exclude cells that have a 1.0 perfect success rate or cells that do not show up in the randomly
        generated test cases.

        :param successMap: Map of perfectly predicted test cases, cell number as keys and list of perfectly predicted frames as values
        :param failedMap: Map of incorrectly predicted test cases, cell number as keys and list of lists containing incorrect predictions and frames as values
        :param nCells: Int number of cells to calculate success rates, set to output size (271 cells) by default, useful when getting metrics for one cell only
        :param excludeMissing: boolean value deciding whether to include cells not tested
        :param excludePerfect: boolean value deciding whether to include cells with 100% accuracy of prediction
        :return: List of cells (str) and list of success rates (float) of the cell in the same index
        """
        successRates = []
        cells = []
        for c in range(self.outputSize):
            totalPred, successRate = self.getCellSuccessRate(c, successMap, failedMap)
            if totalPred > 0:
                successRate = len(successMap.get(c, 0)) / float(totalPred)
                print('total pred', totalPred, 'successes', len(successMap.get(c, 0)), 'success rate', successRate)
                if successRate == 1.0:
                    if excludePerfect:
                        continue
                cells.append(str(c))
                successRates.append(successRate)
            else:
                if not excludeMissing:
                    successRates.append(-1)
                    cells.append(str(c))
                continue
        print("Success rates: ", successRates)
        print("Number of cells: ", len(cells), "Number of success rates: ", len(successRates))
        return cells, successRates


    def plotSuccessRates(self, cells, successRates):
        """
        Plots success rates against cell number using matplotlib.
        :param cells: List of cells, each entry MUST be a string and indices must be aligned with successRates
        :param successRates: List of success rates per cell, indices must be aligned with cells
        :return:
        """
        plt.scatter(cells, successRates)
        plt.xlabel('Cell Number', fontsize=18)
        plt.ylabel('Success Rate %', fontsize=18)
        plt.title('Success Rate Per Cell', fontsize=18)
        plt.show()


    def showImagesNWorstCells(self, cells, successRates, framesMap, successMap, failedMap, frameProbability, bottomN = 5):
        """
        Displays the n photos used in testnImagesEachCell of the bottomN number of cells with the lowest accuracies.
        Each image window displays whether the tested image was successfully or unsuccessfully predicted, the
        predicted cell (if unsuccessful), the probability of the model predicting the actual cell from the image,
        and the probability of the model predicting the wrong cell (if the photo was unsuccessfully predicted).
        Method is meant to be called inside testnImagesAllCells
        :param cells: List of cells tested, indices align with successRates
        :param successRates: List of success rates per cell, indices align with cells
        :param framesMap: Map of n randomly selected frames per cell with cells for keys, list of frames as values
        :param successMap: Map of successful frames predicted per cell with cells for keys, list of frames as values
        :param failedMap: Map of unsuccessfully predicted frames per cell, with cells for keys, list of lists containing
        failed predicted cell and frame number
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities
        for each cell generated by the model as values
        :param bottomN: number of n lowest performing cells
        :return:
        """
        bottomNRate, bottomNCellID = self.findBottomX(bottomN, successRates)
        print(bottomNRate)
        print(bottomNCellID)
        for i in range(bottomN):
            worstSuccessRate = bottomNRate[i]
            indexOfWorstCell = bottomNCellID[i]
            worstCell = int(cells[indexOfWorstCell])

            #get list of nested lists containing [list of failed frames], [list of failed predictions each frame]
            listOfFramesFailedPred = failedMap.get(worstCell, [])

            #create new map with failed frame number as keys, failed prediction per frame as values
            cellFailFramesMap ={list[0]:list[1] for list in listOfFramesFailedPred}

            print('Worst Performing Cell: ', worstCell, " Success Rate: ", worstSuccessRate)
            framesList = framesMap.get(worstCell)
            successFrames = successMap.get(worstCell)

            self.showImagesOneCell(worstCell, framesList, frameProbability, successFrames, cellFailFramesMap)



    def showImagesOneCell(self, cell, framesList, frameProbability, successFrames, cellFailFramesMap):
        """
        Displays the frames corrensponding to one cell used in testing the RGB cell predictor. Each window
        shows information about each frame's prediction result such as predicted cell, prob of prediction,
        prof of actual (if failed prediction), and failed prediction
        :param cell: cell whose corresnponding photos will be shown
        :param framesList: list of frames corresponding to the given cell, each frame will be displayed individually
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each cell generated by the model as values
        :param successFrames: List of successfully predicted frames corresponding to the cell
        :param cellFailFramesMap: Map of unsuccessfully predicted frames, with frame numbers for keys and the failed cell prediction for the frame as values
        :return:
        """
        for frame in framesList:
            imFile = makeFilename(self.frames, frame)
            image = cv2.imread(imFile)
            if image is None:
                print(" image not found")
                continue
            if frame in successFrames:
                probForActualCell = frameProbability[frame][cell]
                cv2.imshow("Frame: " + str(frame) + " Cell " + str(cell) + " Success, Prob Actual: " + str(
                    probForActualCell), image)
            else:
                predCell = cellFailFramesMap.get(frame)
                probForWrongPrediction = frameProbability[frame][predCell]
                probForActualCell = frameProbability[frame][cell]
                cv2.imshow("FR: " + str(frame) + " Fail " + str(cell) + " Pred " + str(predCell) + ", Prob: " + str(
                    probForWrongPrediction) + ", Prob Actual: " + str(probForActualCell), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def logNWorstCells(self, filename, cells, successRates, framesMap, successMap, failedMap, frameProbability, frameTop3PredProb, bottomN = 1):
        """
        Creates a CSV Log to record testing performance metrics (frame number, predicted cell, actual cell, cell success rate, probability of actual,
        probability of predicted, top three predicted cells, top three probabilities) inside res/csvLogs/cellPredictorRGBTestLogs.
        Meant to be called inside testNImagesAllCells for logging purposes.

        :param filename: String filename name that logs will be saved under inside res/csvLogs/cellPredictorRGBTestLogs
        :param cells: List of cells tested, indices align with successRates
        :param successRates: List of success rates per cell, indices align with cells
        :param framesMap: Map of n randomly selected frames per cell with cells for keys, list of frames as values
        :param successMap: Map of successful frames predicted per cell with cells for keys, list of frames as values
        :param failedMap: Map of unsuccessfully predicted frames per cell, with cells for keys, list of lists containing
        failed predicted cell and frame number
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities
        for each cell generated by the model as values
        :param frameTop3PredProb: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Cells]) as values
        :param bottomN: number of n lowest performing cells
        """
        dirTimeStamp = "{}".format(time.strftime("%m%d%y%H%M"))
        logPath = "../../res/csvLogs2022/cellPredictorRGBTestLogs/"
        csvLog = open(logPath + filename + "-" + dirTimeStamp + ".csv", "w")
        filewriter = csv.writer(csvLog)
        filewriter.writerow(["Frame", "Actual Cell", "Predicted Cell", "Success", "Cell Success Rate", "Prob Actual", "Prob Predicted", "Top 3 Pred", "Top 3 Prob"])

        bottomNRate, bottomNCellID = self.findBottomX(bottomN, successRates)
        for i in range(bottomN):
            worstSuccessRate = bottomNRate[i]
            indexOfWorstCell = bottomNCellID[i]
            worstCell = int(cells[indexOfWorstCell])

            listOfList = failedMap.get(worstCell)
            cellFailFramesMap ={list[0]:list[1] for list in listOfList}

            framesList = framesMap.get(worstCell)
            successFrames = successMap.get(worstCell)

            self.logOneCell(filewriter, worstCell, worstSuccessRate, framesList, successFrames, cellFailFramesMap, frameProbability, frameTop3PredProb)
        csvLog.close()


    def logOneCell(self, filewriter, cell, cellSuccessRate, framesList, successFrames, cellFailFramesMap, frameProbability, frameTop3PredProb):
        """
        Takes in a csv filewriter object and logs test performance information of one cell into the csv associated with filewriter.
        Records the frame number, actual cell, predicted cell, cell success rate, probability of prediction, top 3 predictions, and top 3 probabilties
        for the specified cell that was tested. Meant to be called in other test functions for ease of logging. CSV must be closed outside this function
        for logs to be successfully saved.

        :param filewriter: CSV filewriter object associated with the csv log
        :param cell: cell number of cell being tested
        :param cellSuccessRate: success rate of the cell being tested
        :param framesList: list of frame names used in testing that all correspond to cell
        :param successFrames: list of successfully predicted frames belonging to cell
        :param cellFailFramesMap Map of unsuccessfully predicted frames, with frame numbers for keys and the failed cell prediction for the frame as values
        :param frameProbability: Map of probabilities per frame, with frame numbers for keys, and list of probabilities for each cell generated by the model as values
        :param frameTop3PredProb: Map of top 3 predictions and prediction probabilities per frame, with frame numbers for keys and list of lists ([Top Three Probabilities, Top 3 Cells]) as values
        :return:
        """
        for frame in framesList:
            probForActualCell = frameProbability[frame][cell]
            frameTop3Prob = frameTop3PredProb[frame][0]
            frameTop3PredCell = frameTop3PredProb[frame][1]
            if frame in successFrames:
                filewriter.writerow(
                    [frame, str(cell), str(cell), "T", str(cellSuccessRate), str(probForActualCell),
                     str(probForActualCell), str(frameTop3PredCell), str(frameTop3Prob)])
            else:
                predCell = cellFailFramesMap.get(frame)
                probForWrongPrediction = frameProbability[frame][predCell]
                probForActualCell = frameProbability[frame][cell]
                filewriter.writerow(
                    [frame, str(cell), str(predCell), "F", str(cellSuccessRate), str(probForActualCell),
                     str(probForWrongPrediction), str(frameTop3PredCell), str(frameTop3Prob)])


    def graphConfusionMatrix(self, n, true_Label, predict_Label):
        """
        Graphs a confusion matrix using matplotlib. Currently not very informative
        because of the 271x271 size of the confusion matrix.

        :param n: number of tests run/cells predicted
        :param true_Label: List of true cell numbers in order of prediction
        :param predict_Label: List of predicted cell numbers in order of prediction
        :return:
        """
        conf_matrix = np.zeros((self.outputSize, self.outputSize))

        for i in range(n):
            conf_matrix[true_Label[i]][predict_Label[i]] += 1

        print(conf_matrix.shape[0])
        norm = np.linalg.norm(conf_matrix)
        norm_conf_matrix = conf_matrix / norm

        fig = plt.figure(figsize=(self.outputSize,self.outputSize))
        plt.matshow(norm_conf_matrix, fignum=fig.number)

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()


if __name__ == "__main__":
    cellPredictor = CellPredictorRGB(
        # dataSize=95810,
        data_name="TestHeadingInCellPredAdam224",
        checkPointFolder=checkPts,
        imagesFolder=frames,
        imagesParent=DATA + "frames/",
        batch_size=10,
        labelMapFile=DATA + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
        # loaded_checkpoint = "2022CellPredict_checkpoint-0728221645/Test224GlobalPoolingCellPredictorSecond100Epoch-61-1.71.hdf5"
    )

    cellPredictor.buildNetwork()

    #for training

    cellPredictor.prepDatasets()
    cellPredictor.train(epochs = 100)

    #for testing

    #cellPredictor.test(1000)
    # cellPredictor.testnImagesAllCells(100)
    #cellPredictor.testnImagesOneCell(27, 100)
