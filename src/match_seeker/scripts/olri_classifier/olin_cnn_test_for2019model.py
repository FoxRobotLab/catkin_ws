#!/usr/bin/env python3.5

"""--------------------------------------------------------------------------------
olin_cnn_test_for2019model.py
Author: Jinyoung Lim, Avik Bosshardt, Angel Sylvester and Maddie AlQatami
Creation Date: July 2018
Updated: Summer 2019, Summer 2020, Summer 2022

This file contains two main testing functions: testingOnCellOutputNetwork and testingOnHeadingOutputNetwork.
testingOrigData and testingSusanData are no longer used.

The two functions load and test the 2019 model on the existing training dataset.

They preprocess and then predict either the cell number or the heading of each frames, one at a time.

We no longer use the NEWTRAININGDATA_100_500withHeadingInput95k.npy
file as input because it requires the program to load the whole dataset before feeding the frames into the network.



FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes

The OlinClassifier class is imported from olinClassifier.py file!
--------------------------------------------------------------------------------"""

import random
import numpy as np
import cv2

from paths import pathToMatchSeeker, DATA, frames
from imageFileUtils import makeFilename
from preprocessData import DataPreprocess
from olinClassifiers import OlinClassifier



def makeFrameDict(dataFile):
    """Reads the data from the data file, making a dictionary where the keys are the frames numbers, and the data attached is
    another dictionary, with entries for the cell, heading, and location."""
    dataDict = {}
    with open(dataFile) as fp:
        for line in fp:
            ln = line.strip().split(' ')
            entry = int(ln[0])
            dataDict[entry] = {}
            dataDict[entry]['cell'] = int(ln[1])
            dataDict[entry]['heading'] = int(ln[4])
            dataDict[entry]['location'] = (float(ln[2]), float(ln[3]))
    return dataDict



# def check_data(datasetFile):
#     """Reads in the dataset from the given file (should be image data. It randomly shuffles the data, and then does something..."""
#     # TODO: Not sure what this is doing, seems suspicious how it is getting both cell and heading
#     data = np.load(pathToMatchSeeker + 'res/classifier2019Data/DATA/TRAININGDATA_100_500_heading-input_gnrs.npy')
#     np.random.shuffle(data)
#     print(data[0])
#     potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
#     for i in range(len(data)):
#         print("cell:"+str(np.argmax(data[i][1])))
#         print("heading:"+str(potentialHeadings[int(data[i][0][0,0,1])]))
#         cv2.imshow('im',data[i][0][:,:,0])
#         cv2.moveWindow('im',200,200)
#         cv2.waitKey(0)



def cleanImage(image, mean=None, imageSize = 100):
    """Preprocessing the images in similar ways to the training dataset of 2019 model."""
    shrunkenIm = cv2.resize(image, (imageSize, imageSize))
    grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
    meaned = np.subtract(grayed, mean)
    return shrunkenIm, grayed, meaned



def testingOrigData():
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    meanFile = "AngelTRAININGDATA_100_500_mean.npy"
    dataPath = pathToMatchSeeker + 'res/classifier2019data/'
    print("Setting up preprocessor to get frames data...")
    dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
    print("Loading mean...")

    mean = np.load(dataPath + meanFile)
    print(mean.max(), mean.min())
    print(mean)
    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     data_name="cellInput",
                                     headingInput=True,
                                     outputSize=271,
                                     image_size=100,
                                     image_depth=2)

    print("Loading data...")
    dataPackage = np.load(dataPath + cellOutputFile, allow_pickle=True, encoding='latin1')
    print("Data read")
    # print(dataPackage.shape)
    imageData = dataPackage[:, 0]
    cellData = dataPackage[:, 1]
    numIms = imageData.shape[0]
    # print(imageData[0][:, :, 0])
    for i in range(1000):
        #for 1000 images in the dataset, find images
        # in the same cell in NEWTRAININGDATA_100_500withHeadingInput95k.npy
        #and compare them and the predictions the network outputs on them
        print("===========", i)
        imFile = makeFilename(dataPath + 'frames/moreFrames/', i)
        imageB = cv2.imread(imFile)
        cellB = dPreproc.frameData[i]['cell']
        headingB = dPreproc.frameData[i]['heading']
        headingIndex = potentialHeadings.index(headingB)
        print(" cellB =", cellB)
        if imageB is None:
            print(" image not found")
            continue
        smallerB, grayB, processedB = cleanImage(imageB, mean)

        headBArr = headingIndex * np.ones((100, 100, 1))
        print(headingB, headingIndex)
        procBPlus = np.concatenate((np.expand_dims(processedB, axis=-1), headBArr), axis=-1)
        predB = olin_classifier.predictSingleImage(procBPlus)
        cellCount = 0
        for j in range(numIms):
            # if j % 1000 == 0:
            #     print("    ", j)
            cellA = np.argmax(cellData[j])
            if cellA != cellB:
                continue
            cellCount += 1

            imageA = imageData[j][:, :, 0]
            predA = olin_classifier.predictSingleImage(imageData[j])
            # print(imageA[:10, :10])
            diff = np.sum(np.abs(imageA - processedB))
            if diff < 100000:
                print("cellA =", cellA, "cellB =", cellB, "predB =", predB, "predA =", predA)
                print("Difference", diff)
                dispImageA = cv2.convertScaleAbs(imageA)
                dispProcB = cv2.convertScaleAbs(processedB)
                cv2.imshow("Image A", cv2.resize(dispImageA, (400, 400)))
                cv2.moveWindow("ImageA", 50, 500)
                cv2.imshow("Image B", cv2.resize(imageB, (400, 400)))
                cv2.imshow("Smaller B", cv2.resize(smallerB, (400, 400)))
                cv2.imshow("Gray B", cv2.resize(grayB, (400, 400)))
                cv2.imshow("Proce B", cv2.resize(dispProcB, (400, 400)))
                cv2.moveWindow("Proce B", 500, 500)
                cv2.waitKey(5000)



def testingSusanData():
    """Chooses random images from the susantestdataset file and checks them on the network.
    """
    cellOutputData = "susantestdataset.npz"
    # cellOutputImg = "SAMPLETRAININGDATA_IMG_withHeadingInput135K.npy"

    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    # headingOutputData = "SAMPLETRAININGDATA_HEADING_withCellInput135K.npy"
    # headingOutputCheckpoint = "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5"

    dataPath = pathToMatchSeeker + 'res/classifier2019data/'
    # mean = np.load(dataPath + 'SAMPLETRAINING_100_500_mean135k.npy')

    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     data_name="cellInput",
                                     headingInput=True,
                                     outputSize=271,
                                     image_size=100,
                                     image_depth=2)


    npzfile = np.load(dataPath + cellOutputData)
    imageData = npzfile['images']
    cellData = npzfile['cellOut']
    headData = npzfile['headingOut']
    mean = npzfile['mean']
    frameData = npzfile['frameNums']

    imDims = imageData.shape
    cellDims = cellData.shape
    numExamples = imDims[0]
    imageShape = imDims[1:]
    countPerfect = 0
    countTop3 = 0
    countTop5 = 0
    for i in range(1000):
        randRow = random.randrange(numExamples)
        image = imageData[randRow]
        frameNum = frameData[randRow]
        output = cellData[randRow]
        currHeading = np.argmax(headData[randRow])
        expectedResult = np.argmax(output)
        print(frameNum, "heading =", currHeading, "cell =", expectedResult)
        headArr = currHeading * np.ones(imageShape, np.float64)
        predImg = np.dstack((image, headArr))

        # predImg = np.array([image])
        networkResult, allRes = olin_classifier.predictSingleImageAllData(predImg)
        topThreePercsframes, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
        print("Expected result:", expectedResult, "Actual result:", networkResult)
        if networkResult == expectedResult:
            countPerfect += 1
        if expectedResult in topThreeCells:
            countTop3 += 1
        if expectedResult in topFiveCells:
            countTop5 += 1
    print("Count of perfect:", countPerfect)
    print("Count of top 3:", countTop3)
    print("Count of top 5:", countTop5)


def testingOnCellOutputNetwork(n):
    """This runs each of the first n images in the folder of frames through the cell-output network, reporting how
    often the correct cell was produced0, and how often the correct heading was in the top 3 and top 5."""
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    #cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    meanFile = "TRAININGDATA_100_500_mean.npy"
    dataPath = DATA

    print("Setting up preprocessor to get frames data...")
    dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

    print("Loading mean...")
    mean = np.load(dataPath + meanFile)

    print("Setting up classifier loading checkpoints...")
    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     headingInput=True,
                                     outputSize=271,
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
        headingIndex = potentialHeadings.index(headingB)    # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
        smallerB, grayB, processedB = cleanImage(imageB, mean)
        headBArr = headingIndex * np.ones((100, 100, 1))
        procBPlus = np.concatenate((np.expand_dims(processedB, axis=-1), headBArr), axis=-1)
        predB, output = olin_classifier.predictSingleImageAllData(procBPlus)
        topThreePercs, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
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


def testingOnHeadingOutputNetwork(n):
    """This runs each of the first n images in the folder of frames through the heading-output network, reporting how often the correct
    heading was produced, and how often the correct heading was in the top 3 and top 5."""
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    #cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5"
    meanFile = "TRAININGDATA_100_500_mean.npy"
    dataPath = DATA

    print("Setting up preprocessor to get frames data...")
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
        headingIndex = potentialHeadings.index(headingB)    # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
        smallerB, grayB, processedB = cleanImage(imageB, mean)

        cellBArr = cellB * np.ones((100, 100, 1))
        procBPlus = np.concatenate((np.expand_dims(processedB, axis=-1), cellBArr), axis=-1)
        predB, output = olin_classifier.predictSingleImageAllData(procBPlus)
        topThreePercs, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
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
        if val > topVals[x-1]:
            topIndex.insert(j, i)
            topVals.insert(j, val)
            topIndex.pop(-1)
            topVals.pop(-1)
    return topVals, topIndex







# Test random elements of susan's test dataset on network
# testingSusanData()

# Test random elements of original dataset on network
# testingOrigData()

# Run each of the first 1000 images through the cell output trained CNN from 2019
testingOnCellOutputNetwork(1000)

# Run each of the first 1000 images through the heading output trained CNN from 2019
testingOnHeadingOutputNetwork(1000)
