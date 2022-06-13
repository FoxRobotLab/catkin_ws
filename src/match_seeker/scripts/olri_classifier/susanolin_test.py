#!/usr/bin/env python3.5

"""--------------------------------------------------------------------------------
olin_cnn.py
Authors: Susan Fox, Jinyoung Lim, Avik Bosshardt, Angel Sylvester Maddie AlQatami, Arif Zamil, Analeidi Barrera
Creation Date: July 2018
Updated: Summer 2019, Summer 2020

A convolutional neural network to classify 2x2 cells of Olin Rice. Based on
Floortype Classifier CNN, which is based on CIFAR10 tensorflow tutorial
(layer architecture) and cat vs dog kaggle (preprocessing) as guides. Uses
Keras as a framework.

Acknowledgements:
    ft_floortype_classifier
        floortype_cnn.py

Notes:
    Warning: "Failed to load OpenCL runtime (expected version 1.1+)"
        Do not freak out you get this warning. It is expected and not a problem per
        https://github.com/tensorpack/tensorpack/issues/502

    Error: F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn
        tensor descriptor: CUDNN_STATUS_BAD_PARAM. Might occur when feeding an
        empty images/labels

    To open up virtual env:
        source ~/tensorflow/bin/activate

    Use terminal if import rospy does not work on PyCharm but does work on a
    terminal


FULL TRAINING IMAGES LOCATED IN match_seeker/scripts/olri_classifier/frames/moreframes
--------------------------------------------------------------------------------"""


import os
import random
import numpy as np
import cv2
# from tensorflow import keras

from paths import pathToMatchSeeker, DATA
from imageFileUtils import makeFilename
from frameCellMap import FrameCellMap
from olinClassifiers import OlinClassifier


def cleanImage(image, mean=None, imageSize = 100):
    """Converts the image to the form used when saving the data."""
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
    dPreproc = FrameCellMap(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
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
    # imageData = np.load(dataPath +  cellOutputImg, allow_pickle=True, encoding='latin1')
    # cellData = np.load(dataPath +  cellOutputData, allow_pickle=True, encoding='latin1')

    imDims = imageData.shape
    cellDims = cellData.shape
    numExamples = imDims[0]
    imageShape = imDims[1:]
    count = 0
    for i in range(1000):
        randRow = random.randrange(numExamples)
        image = imageData[randRow]
        frameNum = frameData[randRow]
        output = cellData[randRow]
        currHeading = np.argmax(headData[randRow])
        expectedResult = np.argmax(output)
        print(frameNum, "heading =", currHeading, "cell =", expectedResult)
        headArr = currHeading * np.ones(imageShape, np.float64)
        print(currHeading)
        image = np.dstack((image, headArr))

        print()
        predImg = np.array([image])
        networkResult = olin_classifier.predictSingleImage(predImg)
        print("Expected result:", expectedResult, "Actual result:", networkResult)
        if expectedResult == networkResult:
            count += 1
    print("Number correct =", count)


def testingOnCellOutputNetwork(n):
    """This runs each of the first n images in the folder of frames through the cell-output network, reporting how
    often the correct cell was produced, and how often the correct heading was in the top 3 and top 5."""
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    meanFile = "AngelTRAININGDATA_100_500_mean.npy"
    dataPath = pathToMatchSeeker + 'res/classifier2019data/'

    print("Setting up preprocessor to get frames data...")
    dPreproc = FrameCellMap(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

    print("Loading mean...")
    mean = np.load(dataPath + meanFile)

    print("Setting up classifier loading checkpoints...")
    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     data_name="cellInput",
                                     headingInput=True,
                                     outputSize=271,
                                     image_size=100,
                                     image_depth=2)

    countPerfect = 0
    countTop3 = 0
    countTop5 = 0
    for i in range(n):
        print("===========", i)
        imFile = makeFilename(dataPath + 'frames/moreFrames/', i)
        imageB = cv2.imread(imFile)
        cellB = dPreproc.frameData[i]['cell']
        headingB = dPreproc.frameData[i]['heading']
        headingIndex = potentialHeadings.index(headingB)    # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
        if imageB is None:
            print(" image not found")
            continue
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
        x = cv2.waitKey(500)
        if chr(x & 0xFF) == 'q':
            break
    print("Count of perfect:", countPerfect)
    print("Count of top 3:", countTop3)
    print("Count of top 5:", countTop5)


def testingOnHeadingOutputNetwork(n):
    """This runs each of the first n images in the folder of frames through the heading-output network, reporting how often the correct
    heading was produced, and how often the correct heading was in the top 3 and top 5."""
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5"
    meanFile = "AngelTRAININGDATA_100_500_mean.npy"
    dataPath = pathToMatchSeeker + 'res/classifier2019data/'

    print("Setting up preprocessor to get frames data...")
    dPreproc = FrameCellMap(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")

    print("Loading mean...")
    mean = np.load(dataPath + meanFile)

    print("Setting up classifier loading checkpoints...")
    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     data_name="cellInput",
                                     headingInput=True,
                                     outputSize=271,
                                     image_size=100,
                                     image_depth=2)
    countPerfect = 0
    countTop3 = 0
    countTop5 = 0
    for i in range(n):
        print("===========", i)
        imFile = makeFilename(dataPath + 'frames/moreFrames/', i)
        imageB = cv2.imread(imFile)
        cellB = dPreproc.frameData[i]['cell']
        headingB = dPreproc.frameData[i]['heading']
        headingIndex = potentialHeadings.index(headingB)    # This is what was missing. This is converting from 0, 45, 90, etc. to 0, 1, 2, etc.
        if imageB is None:
            print(" image not found")
            continue
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
        x = cv2.waitKey(500)
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





# This tedious function compared each picture from the first 1000 (what I have on my computer) to every picture in the
# dataset, and if they were (1) in the same cell and (2) were similar enough, then they were displayed, along with the
# actual and predicted cell numbers
# testingOrigData()

# Run each of the first 1000 images through the cell output trained CNN from 2019
# testingOnCellOutputNetwork(1000)

# Run each of the first 1000 images through the heading output trained CNN from 2019
testingOnHeadingOutputNetwork(1000)
