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
import numpy as np
# from tensorflow import keras
import cv2
# import time
from paths import pathToMatchSeeker, DATA
from imageFileUtils import makeFilename
from preprocessData import DataPreprocess
# ORIG import olin_inputs_2019 as oi2
import random

from olinClassifiers import OlinClassifier

### Uncomment next line to use CPU instead of GPU: ###
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

#
# def makeFilename(path, fileNum):
#     """Makes a filename for reading or writing image files"""
#     formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
#     name = formStr.format(path, 'frame', fileNum, "jpg")
#     return name
#
#
# def getImageFilenames(path):
#     """Read filenames in folder, and keep those that end with jpg or png  (from copyMarkedFiles.py)"""
#     filenames = os.listdir(path)
#     keepers = []
#     for name in filenames:
#         if name.endswith("jpg") or name.endswith("png"):
#             keepers.append(name)
#     return keepers
#
#
# def extractNum(fileString):
#     """Finds sequence of digits"""
#     numStr = ""
#     foundDigits = False
#     for c in fileString:
#         if c in '0123456789':
#             foundDigits = True
#             numStr += c
#         elif foundDigits:
#             break
#     if numStr != "":
#         return int(numStr)
#     else:
#         return -1
#

def loading_bar(start,end, size = 20):
    # Useful when running a method that takes a long time
    loadstr = '\r'+str(start) + '/' + str(end)+' [' + int(size*(float(start)/end)-1)*'='+ '>' + int(size*(1-float(start)/end))*'.' + ']'
    if start % 10 == 0:
        print(loadstr)



def cleanImage(image, mean=None, imageSize = 100):
    """Converts the image to the form used when saving the data."""
    shrunkenIm = cv2.resize(image, (imageSize, imageSize))
    grayed = cv2.cvtColor(shrunkenIm, cv2.COLOR_BGR2GRAY)
    meaned = np.subtract(grayed, mean)
    return shrunkenIm, grayed, meaned



# def clean_image_old(image, data = 'old', cell = None, heading = None):
#     #mean = np.load(pathToMatchSeeker + 'res/classifier2019data/TRAININGDATA_100_500_mean95k.npy')
#     image_size = 100
#     if data == 'old': #compatible with olin_cnn 2018
#         resized_image = cv2.resize(image, (image_size, image_size))
#         gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#         image = np.subtract(gray_image, mean)
#         depth = 1
#     elif data == 'vgg16': #compatible with vgg16 network for headings
#         image = cv2.resize(image, (170, 128))
#         x = random.randrange(0, 70)
#         y = random.randrange(0, 28)
#         image = image[y:y + 100, x:x + 100]
#         depth = 3
#     elif data == 'cell_channel':
#         if cell != None:
#             resized_image = cv2.resize(image, (image_size, image_size))
#             gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#             image = np.subtract(gray_image, mean)
#             cell_arr = cell * np.ones((image_size, image_size, 1))
#             image = np.concatenate((np.expand_dims(image,axis=-1),cell_arr),axis=-1)
#             depth = 2
#         else:
#             print("No value for cell found")
#     elif data == 'heading_channel':
#         if heading != None:
#             resized_image = cv2.resize(image, (image_size, image_size))
#             gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#             image = np.subtract(gray_image, mean)
#             cell_arr = heading * np.ones((image_size, image_size, 1))
#             image = np.concatenate((np.expand_dims(image,axis=-1), cell_arr),axis=-1)
#             depth = 2
#         else:
#             print("No value for heading found")
#     else: #compatible with olin_cnn 2019
#         image = cv2.resize(image, (170, 128))
#         x = random.randrange(0, 70)
#         y = random.randrange(0, 28)
#         cropped_image = image[y:y + 100, x:x + 100]
#         gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#         image = np.subtract(gray_image, mean)
#         depth = 1
#     cleaned_image = np.array([image], dtype="float") \
#         .reshape(1, image_size, image_size, depth)
#     return cleaned_image
#



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


def testingOnNetwork():
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    meanFile = "AngelTRAININGDATA_100_500_mean.npy"
    dataPath = pathToMatchSeeker + 'res/classifier2019data/'
    print("Setting up preprocessor to get frame data...")
    dPreproc = DataPreprocess(dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
    print("Loading mean...")

    mean = np.load(dataPath + meanFile)
    checkPts = dataPath + "CHECKPOINTS/"
    olin_classifier = OlinClassifier(checkpoint_dir=checkPts,
                                     savedCheckpoint=checkPts + cellOutputCheckpoint,
                                     data_name="cellInput",
                                     headingInput=True,
                                     outputSize=271,
                                     image_size=100,
                                     image_depth=2)

    # print("Loading data...")
    # dataPackage = np.load(dataPath + cellOutputFile, allow_pickle=True, encoding='latin1')
    # print("Data read")
    # print(dataPackage.shape)
    # imageData = dataPackage[:, 0]
    # cellData = dataPackage[:, 1]
    # numIms = imageData.shape[0]
    # print(imageData[0][:, :, 0])
    countPerfect = 0
    countTop3 = 0
    countTop5 = 0

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
        predB, output = olin_classifier.predictSingleImageAllData(procBPlus)
        topThreePercs, topThreeCells = findTopX(3, output)
        topFivePercs, topFiveCells = findTopX(5, output)
        print("cellB =", cellB, "   predB =", predB)
        print("Top three:", topThreeCells, topThreePercs)
        print("Top five:", topFiveCells, topFivePercs)
        if predB == cellB:
            print("perfect!")
            countPerfect += 1
        if cellB in topThreeCells:
            print("top 3")
            countTop3 += 1
        if cellB in topFiveCells:
            print("top 5")
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


def findTopX(x, cellArray):
    topVals = [0.0] * x
    topIndex = [None] * x
    for i in range(len(cellArray)):
        val = cellArray[i]

        for j in range(x):
            if topIndex[j] is None or val > topVals[j]:
                break
        if val > topVals[x-1]:
            topIndex.insert(j, i)
            topVals.insert(j, val)
            topIndex.pop(-1)
            topVals.pop(-1)
    return topVals, topIndex






def testingOrigData():
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    cellOutputFile = "NEWTRAININGDATA_100_500withHeadingInput95k.npy"
    cellOutputCheckpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5"
    meanFile = "AngelTRAININGDATA_100_500_mean.npy"
    dataPath = pathToMatchSeeker + 'res/classifier2019data/'
    print("Setting up preprocessor to get frame data...")
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

#
# testingOrigData()
testingOnNetwork()
