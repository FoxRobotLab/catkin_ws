"""
Data generator based on DataGenerator2022.py that produces data suitable for use by a CNN-LSTM model. It
produces sequences of a fixed length, made from overlapping segments of each video/run frames.

Created Summer 2024
Authors: Susan Fox, Marcus Wallace, Elisa Avalos, Oscar Reza Bautista
"""

import os
import numpy as np
from tensorflow import keras
import cv2

import math
import re

# Store the long path to access the data folders
myPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data/DATA/"
textDataPath = myPath + "AnnotData/"
framesDataPath = myPath + "FrameData/classifier2022Data/DATA/FrameData/"
checkPts = myPath + "CHECKPOINTS/"


class DataGeneratorLSTM(keras.utils.Sequence):
    def __init__(self, framePath, annotPath, skipSize = 3, seqLength = 5,
                 batch_size=20, shuffle=True, randSeed=12342, train_perc=0.2,
                 img_size=224, train=True, generateForCellPred = True):

        self.batch_size = batch_size
        self.framePath = framePath
        self.annotPath = annotPath
        self.skipSize = skipSize
        self.seqLength = seqLength

        self.shuffle = shuffle
        self.img_size = img_size
        self.image_path = framesDataPath

        self.runData = self._collectRunData()
        self.allSequences = self._enumerateSequences()
        print(len(self.allSequences))

        self.train_perc = train_perc
        np.random.seed(randSeed)  # set random generator to

        self.trainSequences, self.valSequences = self.traintestsplit(self.allSequences, self.train_perc)
        print(self.train_perc, len(self.trainSequences) / len(self.allSequences), len(self.valSequences) / len(self.allSequences))

        self.generateForCellPred = generateForCellPred


        self.potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315]
        if train:
            self.usedSequences = self.trainSequences
        else:
            self.usedSequences = self.valSequences

    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.usedSequences) / self.batch_size))

    def _collectRunData(self):
        """Iterates over the annotation files in the input folder, and builds a run data object for each."""
        if not os.path.exists(self.annotPath):
            raise FileNotFoundError
        annotFiles = [f for f in os.listdir(self.annotPath) if f.startswith("FrameDataReviewed") and f.endswith(".txt")]
        annotFiles.sort()
        runData = []
        for file in annotFiles:
            nextRunInfo = VideoRunData(file, self.annotPath, self.framePath, self.skipSize, self.seqLength)
            runData.append(nextRunInfo)
        return runData

    def _enumerateSequences(self):
        """Iterates over the self.runData objects, and collects up how many sequences each run has. It makes a list of all
        possible sequences, which can be reordered to produce sequences in random orders."""
        allSequences = []
        for (runIndex, rData) in enumerate(self.runData):
            numSeqs = rData.getNumSequences()
            for seqInd in range(numSeqs):
                allSequences.append([runIndex, seqInd])
        return allSequences

    def on_epoch_end(self):
        'Reshuffles after each epoch'
        if self.shuffle:
            np.random.shuffle(self.usedSequences)

    def __getitem__(self, index):
      """Generate one batch of data"""

      # Generate data
      X, Y = self.__data_generation(index * self.batch_size)
      return X, Y

    def cleanImage(self, image):
        """Preprocessing the images into the correct input form."""
        img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shrunkenIm = cv2.resize(img2, (self.img_size, self.img_size))
        processedIm = shrunkenIm / 255.0
        return processedIm

    def __data_generation(self, startIndex):
        """Generates and returns one batch of data, a batch of sequences, and their corresponding labels.
        Note the shape of the data here: one row for each sequence, and the second dimension is the length
        of the sequence (each element is an image), and the third, fourth, and fifth dimensions are the image size
        (height, width, channels)."""

        # # Initialization
        X = np.zeros((self.batch_size, self.seqLength, self.img_size, self.img_size, 3), dtype=float)
        Y = np.zeros((self.batch_size), dtype=int)

        for bInd in range(self.batch_size):
            actInd = startIndex + bInd
            [runInd, seqInd] = self.usedSequences[actInd]
            runObj = self.runData[runInd]
            frameList, annotList = runObj.retrieveSequence(seqInd)
            for (i, imgName) in enumerate(frameList):
                img = cv2.imread(imgName)
                finalIm = self.cleanImage(img)
                X[bInd,i,:,:,:] = finalIm
            if self.generateForCellPred:
                Y[bInd] = annotList[-1]['cell']
            else:
                headVal = annotList[-1]['head']
                headInd = self.potentialHeadings.index(headVal)
                Y[bInd] = headInd
        return X, Y
    def traintestsplit(self, sequences, train_perc):
        '''Split the data passed in based on the evaluation ratio into
        training and testing datasets, assuming it's already randomized'''
        np.random.shuffle(sequences)
        num_eval = int((train_perc * len(sequences)))
        train_images = sequences[:-num_eval]
        eval_images = sequences[-num_eval:]
        return train_images, eval_images


class VideoRunData(object):
    """Represents the data for one "run" (essentially one video) including the annotations and the frames themselves,
    without reading in the image data. It can be used to retrieve a sequence of image names and their annotations
    of a given length and starting point."""
    def __init__(self, annotFile, annotPath, dataPath, skipSize, seqLength):
        """Sets up the data for a single run, given the annotation filename and the path to the folder of images.
        It also takes optionally the number of frames to skip between starts of sequences, and the length of the
        sequence to produce."""
        # Sequence basics
        self.skipSize = skipSize
        self.seqLength = seqLength

        # identifying timestamp information
        results = re.findall("FrameDataReviewed(\d+)-(\d+)frames.txt", annotFile)
        [date, recTime] = results[0]
        self.date = date
        self.recTime = recTime

        # Set up information about images and their filenames
        self.folderPath = dataPath + str(date) + "-" + str(recTime) + "frames"

        if not os.path.exists(self.folderPath):
            print( "THERE IS NO " + self.folderPath + ("!!!"))
            raise FileNotFoundError
        self.imageNames = [f for f in os.listdir(self.folderPath) if (f.endswith(".jpg") and f.startswith("frame202"))]  # Added the 'and' operator to prevent ._ files to get selected
        self.imageNames.sort()
        self.frameCount = len(self.imageNames)
        self.numSequences = math.ceil((self.frameCount - self.seqLength + 1) / self.skipSize)

        # Set up information about locations and cells from the annotation file
        self.annotationsFile = annotPath + annotFile
        if not os.path.exists(self.annotationsFile):
            raise FileNotFoundError
        with open(self.annotationsFile, 'r') as fil:
            rawLines = fil.readlines()
        self.annotData = {}
        for line in rawLines:
            parts = line.split()
            if len(parts) > 1:  # Checks if the line is not empty
                imgName = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                cell = int(parts[3])
                head = int(parts[4])
                self.annotData[imgName] = {'x': x, 'y': y, 'cell': cell, 'head': head}

    def getFrameCount(self):
        """Returns the number of frames in this run"""
        return self.frameCount

    def getNumSequences(self):
        """Calculates the number of sequences, given the number of frames, the skip size, and the length of the sequence"""
        return self.numSequences

    def retrieveSequence(self, seqInd):
        """This takes in a number, which is NOT the index of the frame, but rather which sequence to retrieve. For
        example, with a skip size of 2, 0 would start with frame 0, but sequence index 1 would start with frame 2.
        It returns a list of the image filenames for this sequence, and the corresponding annotations
        (x, y, cell, head)."""
        startIndex = seqInd * self.skipSize
        seqFramePaths = []
        seqAnnotations = []
        for i in range(self.seqLength):
            imName = self.imageNames[startIndex + i]
            annot = self.annotData[imName]
            seqFramePaths.append(self.folderPath + "/" + imName)
            seqAnnotations.append(annot)
        return seqFramePaths, seqAnnotations



def testingCalcOfSeqs():
    for length in range(10, 26):
        print("----------------------------")
        for skipSize in range(1, 4):
            for seqLen in range(1, 5):
                print("---", length, skipSize, seqLen)
                ll = length - seqLen + 1
                calcSeqNum = math.ceil(ll / skipSize)
                cnt = 0
                for start in range(0, ll, skipSize):
                    outStr = ""
                    for i in range(seqLen):
                        outStr += str(start + i) + " "
                    # print(cnt, ":", outStr)
                    cnt += 1
                # print("Count of sequences", cnt)
                if calcSeqNum != cnt:
                    print("Calculated and count inconsistent:", calcSeqNum, cnt)


if __name__ == "__main__":
    # print(DATA2022)
    # print(DATA2022 + "DATA/FrameData/")
    dataGen = DataGeneratorLSTM(framesDataPath, textDataPath, skipSize=3, seqLength=5)
    X, Y = dataGen[0]
    (b, s, h, w, d) = X.shape
    print(X.shape, Y.shape)
    for i in range(b):
        for j in range(s):
            print("seq = ", i, 'frame =', j)
            cv2.imshow("Test", X[i,j])
            cv2.waitKey(0)