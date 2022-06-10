
"""--------------------------------------------------------------------------------
olin_cnn_test_for2019model.py
Author: Yifan Wu, Shosuke Noma, Bea Bautista
Creation Date: June 2022

This file contains functions that load and test the 2020 model on the existing training dataset.

They preprocess and then predict either the cell number or the heading of each frame, one at a time.

FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes

The OlinClassifier class is imported from olin_cnn.py file!
--------------------------------------------------------------------------------"""

import random
import numpy as np
import cv2

from paths import pathToMatchSeeker, DATA, frames, checkPts
from imageFileUtils import makeFilename
from preprocessData import DataPreprocess
from olin_cnn import OlinClassifier

def cleanImage(image, mean=None, imageSize = 100):
    """Preprocessing the images in similar ways to the training dataset of 2019 model."""
    shrunkenIm = cv2.resize(image, (imageSize, imageSize))
    grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
    meaned = np.subtract(grayed, mean)
    #print(meaned.shape)
    meaned = np.expand_dims(meaned,2)
    #print(meaned.shape)
    return shrunkenIm, grayed, meaned

def testingOnHeadingOutputNetwork(n):
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    meanFile = "TRAININGDATA_100_500_mean.npy"
    dataPath = DATA
    print("Setting up preprocessor to get frame data...")
    dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

    #print("Loading mean...")
    mean = np.load(dataPath + meanFile)

    olin_classifier = OlinClassifier(model2020="Heading") #loads 2020 model
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

        smallerB, grayB, processedB = cleanImage(imageB, mean)
        #processedB = np.expand_dims(grayB, 2)
        predB, output = olin_classifier.predictSingleImageAllData(processedB)
        topThreePercs, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
        headingB = dPreproc.frameData[rand]['heading']
        headingIndex = potentialHeadings.index(headingB)  # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
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
        # cv2.imshow("Smaller B", cv2.resize(smallerB, (400, 400)))
        # cv2.moveWindow("Smaller B", 50, 500)
        # cv2.imshow("Gray B", cv2.resize(grayB, (400, 400)))
        # cv2.moveWindow("Gray B", 500, 500)
        cv2.imshow("Proce B", cv2.resize(dispProcB, (400, 400)))
        cv2.moveWindow("Proce B", 500, 50)
        x = cv2.waitKey(50)
        if chr(x & 0xFF) == 'q':
            break
    print("Count of perfect:", countPerfect)
    print("Count of top 3:", countTop3)
    print("Count of top 5:", countTop5)

def testingOnCellOutputNetwork(n):
    meanFile = "TRAININGDATA_100_500_mean.npy"
    dataPath = DATA
    print("Setting up preprocessor to get frame data...")
    dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

    #print("Loading mean...")
    mean = np.load(dataPath + meanFile)

    olin_classifier = OlinClassifier(model2020="Cell") #loads 2020 model
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
        smallerB, grayB, processedB = cleanImage(imageB, mean)
        #processedB = np.expand_dims(grayB, 2)
        predB, output = olin_classifier.predictSingleImageAllData(processedB)
        topThreePercs, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
        cellB = dPreproc.frameData[rand]['cell']
        print("cellB =", cellB, "   predB =", predB)
        print("Top three:", topThreeCells, topThreePercs)
        print("Top five:", topFiveCells, topFivePercs)
        if predB == cellB:
            countPerfect += 1
        if cellB in topThreeCells:
            countTop3 += 1
        if cellB in topFiveCells:
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

def findTopX(x, numList):
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

#Run random n images through CNN from 2020
testingOnCellOutputNetwork(100)
testingOnHeadingOutputNetwork(100)
