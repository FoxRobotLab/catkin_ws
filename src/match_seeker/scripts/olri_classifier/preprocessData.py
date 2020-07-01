"""
File: preprocessData.py
Created: June 2020
Authors: Susan Fox

This file is a new version of olin_2019_inputs.py intended to clean up the code, introduce a class to collect up
a bunch of globalish variables, and make it more efficient. Also trying to fix bugs inadvertently introduced when we
modified the code, so now we can't generate the right kind of datasets.  Will also include an alternative using
a generator, eventually.
TODO: Create a generator version
"""


import cv2
import numpy as np
import os
import random
from datetime import datetime
from paths import DATA
from imageFileUtils import makeFilename


class DataPreprocess(object):


    def __init__(self, imageSize=100, imagesPerCell=500, numOutputs=271, imageDir="", dataFile=""):
        self.imageSize = imageSize
        self.imagesPerCell = imagesPerCell
        self.numOutputs = numOutputs
        self.imageDir = imageDir
        self.dataFile = dataFile

        self.cellsTooFewImages = []
        self.cellsEnoughImages = []
        self.frameData = {}
        self.cellData = {}
        self.headingData = {}
        self.buildDataDicts()


    def buildDataDicts(self):
        """
        Reads in the data in the self.dataFile file, and fills in various dictionaries.
        self.frameData uses the frame number as the key and contains a dictionary with keys 'cell', 'heading', 'loc'
        self.cellData uses the cell number as the key, and the frame number as the value
        self.headingData uses the heading number as the key, and the frame number as the value
        :return: nothing
        """
        with open(self.dataFile) as frameData:
            lines = frameData.readlines()

        for line in lines:
            splitList = line.split()
            frameNum = int(splitList[0])
            cellNum = int(splitList[1])
            xVal = float(splitList[2])
            yVal = float(splitList[3])
            headingNum = int(splitList[4])
            loc = (xVal, yVal)
            self.frameData[frameNum] = {'cell': cellNum, 'heading': headingNum, 'loc': loc}
            if cellNum not in self.cellData:
                self.cellData[cellNum] = [frameNum]
            else:
                cellList = self.cellData[cellNum]
                cellList.append(frameNum)
            if headingNum not in self.headingData:
                self.headingData[headingNum] = [frameNum]
            else:
                headingList = self.headingData[headingNum]
                headingList.append(frameNum)


    def splitCellsByThreshold(self):
        """
        Determines which cells are below the threshold for "enough" frames per cell, versus those who are sufficiently or
        "over" represented. Saves the values in instance variables
        :return: Nothing
        """
        tooFew = []
        enough = []

        for cell in self.cellData:
            numFrames = len(self.cellData[cell])
            if numFrames <= self.imagesPerCell:
                tooFew.append(cell)
            else:
                enough.append(cell)
        self.cellsTooFewImages = tooFew
        self.cellsEnoughImages = enough


    def selectEnoughFrames(self):
        """
        Looks at each cell that has enough frames, and chooses frames for each heading, choosing from each heading in
        turn until headings run out of frames.
        :return: list of frames selected so that we have self.imagesPerCell frames for each cell (for all cells that
        have at least self.imagesPerCell frames recorded.
        """
        chosenFrames = []

        for cell in self.cellsEnoughImages:
            framesForCell = []
            framesEachHeading = {}

            # Build a list for each heading of all frames for this cell that are for this heading
            for fr in self.cellData[cell]:
                frHead = self.frameData[fr]['heading']
                if frHead in framesEachHeading:
                    framesEachHeading[frHead].append(fr)
                else:
                    framesEachHeading[frHead] = [fr]
            frameLists = list(framesEachHeading.values())
            fListLens = [len(l) for l in frameLists]
            listInd = 0

            # loops through each heading index, and randomly selects a frame from that heading list, skipping []
            while len(framesForCell) < self.imagesPerCell:
                if len(frameLists[listInd]) > 0:
                    pickedFrame = random.choice(frameLists[listInd])
                    frameLists[listInd].remove(pickedFrame)
                    fListLens[listInd] -= 1
                    framesForCell.append(pickedFrame)
                listInd = (listInd + 1) % len(frameLists)
            chosenFrames += framesForCell
        return chosenFrames

    def createMoreFrames(self):
        """
        Looks at each cell that has enough frames, and chooses frames for each heading, choosing from each heading in
        turn until headings run out of frames.
        :return: list of frames selected so that we have self.imagesPerCell frames for each cell (for all cells that
        have at least self.imagesPerCell frames recorded.
        """
        chosenFrames = []
        newFrames = []

        for cell in self.cellsTooFewImages:
            print("====================== Cell =", cell, " ==========================")
            oldFramesForCell = []
            newFramesForCell = []
            framesEachHeading = {}

            # Build a list for each heading of all frames for this cell that are for this heading
            for fr in self.cellData[cell]:
                oldFramesForCell.append(fr)
                frHead = self.frameData[fr]['heading']
                if frHead in framesEachHeading:
                    framesEachHeading[frHead].append(fr)
                else:
                    framesEachHeading[frHead] = [fr]
            frameLists = list(framesEachHeading.values())
            fListLens = [len(l) for l in frameLists]

            # loops through each heading index, and randomly selects a frame from that heading list, skipping []
            while len(oldFramesForCell) + len(newFramesForCell) < self.imagesPerCell:
                print(len(oldFramesForCell), len(newFramesForCell))
                print("lengths:", fListLens)
                minInd = np.argmin(fListLens)
                print("index=", minInd)
                if len(frameLists[minInd]) > 0:
                    pickedFrame = random.choice(frameLists[minInd])
                    fListLens[minInd] += 1
                    newFramesForCell.append(pickedFrame)
            chosenFrames += oldFramesForCell
            newFrames += newFramesForCell
        return chosenFrames, newFrames


    def makeOneHotList(self, index, size):
        """
        Constructs a "one-hot" list that is all zeros, except for the position given by the index, which is one.
        :param index: the position in the one-hot list that should be one
        :param size:  the length of the one-hot list
        :return: a list of length size, all zeros, except for a 1 at position index
        """
        onehot = [0] * size
        onehot[index] = 1
        return onehot


    def calculateMean(self, images):
        """
        Takes a set of images, and computes a pixel-wise average brightness or color. The resulting matrix is returned
        :param images: A list of images, could be grayscale, color, or grayscale with a second channel added
        :return: a mean image the same shape as one of the list of images, pixel-wise average brightness or color
        """
        dimens = images[0].shape

        if (len(dimens) == 2) or ((len(dimens) == 3) and (dimens[2] == 3)):  # this is a grayscale image or a color image
            N = 0
            mean = np.zeros(dimens, np.float64)
            for img in images:
                mean += img
                N += 1
            mean /= N
            return mean
        elif (len(dimens) == 3) and (dimens[2] == 2):  # this is a gray image combined with an extra channel
            N = 0
            mean = np.zeros((dimens[0], dimens[1]), np.float64)
            for img in images:
                mean += img[:,:,0]
                N += 1
            mean /= N
            return mean
        else:
            return 0.0


    def readImage(self, frameNum):
        """
        Takes in a frame number, and reads in and returns the image with that frame number.
        :param frameNum: The number of the frame
        :return: Returns an image array (or None if no image with that number)
        """
        fName = makeFilename(self.imageDir, frameNum)
        image = cv2.imread(fName)
        return image


    def resizeAndGray(self, image, imSize=None):
        """
        Takes in an image, and resizes it to be square, and the size for this object. Then it converts it to grayscale
        :param image: Image to be modified
        :return: A new image, scaled to be square and turned to grayscale
        """
        if imSize is None:
            imSize = self.imageSize
        resizedImage = cv2.resize(image, (imSize, imSize))
        grayResizedImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        return grayResizedImage



    def randEraseImage(self, image, minSize=0.02, maxSize=0.4, minRatio=0.3, maxRatio=1 / 0.3, minVal=0, maxVal=255):
        """
        Randomly erase a rectangular part of the given image in order to augment data. Based on code from github:
        https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
        :param image: The image to be modified
        :param minSize: The smallest size of the rectangle to be erased, in pixels
        :param maxSize: The largest size of the rectangle to be erased, in pixels
        :param minRatio: Minimum scaling factor used for the width versus height of rectangle to erase
        :param maxRatio: Maximum scaling factor used for the width versus height of rectangle to erase
        :param minVal: minimum brightness value to set erased rectangle to
        :param maxVal: maximum brightness value to set erased rectangle to
        :return: Returns a new image, with a randomly placed and sized rectangle of a random color placed on it
        """
        global lc, tr, rc, br
        assert len(image.shape) == 2
        reImage = image.copy()
        brightness = np.random.uniform(minVal, maxVal)
        h, w = reImage.shape

        size = np.random.uniform(minSize, maxSize) * h * w
        ratio = np.random.uniform(minRatio, maxRatio)
        width = int(np.sqrt(size / ratio))
        height = int(np.sqrt(size * ratio))

        midRow = np.random.randint(0, h)
        midCol = np.random.randint(0, w)
        topRow = midRow - (height // 2)
        leftCol = midCol - (width // 2)
        reImage[topRow:topRow + height, leftCol:leftCol + width] = brightness
        if topRow < 0:
            tr += 1
        if leftCol < 0:
            lc += 1
        if (topRow + height) >= h:
            br += 1
        if (leftCol + width) >= w:
            rc += 1
        print(tr, br, lc, rc)
        return reImage




if __name__ == "__main__":
    preProc = DataPreprocess(imageDir=DATA + "frames/moreframes/",
                             dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
    # for frame in preProc.frameData:
    #     print("frame:", frame, preProc.frameData[frame])
    preProc.splitCellsByThreshold()

    enoughFrames = preProc.selectEnoughFrames()
    shortFrames, extraFrames = preProc.createMoreFrames()
    lc = 0
    tr = 0
    rc = 0
    br = 0

    # for frame in enoughFrames:
    #     nextIm = preProc.readImage(frame)
    #     if nextIm is None:
    #         continue
    #     sizeGray = preProc.resizeAndGray(nextIm)
    #     gray2 = preProc.resizeAndGray(nextIm, imSize=500)
    #     randErIm = preProc.randEraseImage(gray2)
    #     cv2.imshow("Original", nextIm)
    #     cv2.imshow("Rand erase", randErIm)
    #     cv2.imshow("Smaller", sizeGray)
    #     x = cv2.waitKey(250)
    #     if chr(x & 0xFF) == 'q':
    #         break




def add_cell_channel(allLabels = None, randStart= None, cellInput = None, headingInput=None ):
    """
    This builds something, no idea really
    :param allLabels:
    :param randStart:
    :param cellInput:
    :param headingInput:
    :return:
    """
    frame_cell_dict = getFrameCellDict()
    frame_heading_dict = getFrameHeadingDict()
    train_imgWCell = []
    hotLabelHeading = []
    train_imgWHeading =[]
    hotLabelCell= []
    allImages = []


    if allLabels is None:
        allLabels, randStart = getLabels()

    def processFrame(frame):
        print( "Processing frame " + str(frameNum) + " / " + str(len(allLabels)) + "     (Frame number: " + frame + ")")
        image = cv2.imread(DATA +'frames/moreframes/frame' + frame + '.jpg')
        image = resizeAndCrop(image)
        allImages.append(image)
        return image

    frameNum = 1
    for frame in allLabels:
        img = processFrame(frame)

        if (frameNum -1) >= randStart:
            img = randerase_image(img, 1)
        if(cellInput == True):
            train_imgWCell.append(img)
            hotLabelHeading.append(getOneHotLabel(int(frame_heading_dict[frame]) // 45, 8))

        if(headingInput == True):
            train_imgWHeading.append(img)
            hotLabelCell.append(getOneHotLabel(int(frame_cell_dict[frame]), numCells))

        frameNum += 1

    mean = calculate_mean(allImages)
    #loading_bar(frameNum, len(overRepped) + len(underRepped) + len(randomUnderRepSubset), 150)

    def whichTrainImg():
        if len(train_imgWCell) > len(train_imgWHeading):
            return train_imgWCell
        else:
            return train_imgWHeading

    train_img = whichTrainImg()
    print("expecting a one", len(train_img))

    for i in range(len(train_img)):
        frame = allLabels[i]
        image = train_img[i]
        image = image - mean
        image /= 255
        image = np.squeeze(image)
        if cellInput == True:
            cell = int(frame_cell_dict[frame])
            cell_arr = cell * np.ones((image.shape[0], image.shape[1], 1))
            train_imgWCell[i] = np.concatenate((np.expand_dims(image, axis=-1), cell_arr), axis=-1)

        if headingInput == True:
            heading = (int(frame_heading_dict[frame])) // 45
            heading_arr = heading*np.ones((image.shape[0], image.shape[1], 1))
            train_imgWHeading[i]= np.concatenate((np.expand_dims(image,axis=-1),heading_arr),axis=-1)



    if cellInput == True:
        train_imgWCell = np.asarray(train_imgWCell)
        hotLabelHeading = np.asarray(hotLabelHeading)
        np.save(DATA + 'SAMPLETRAININGDATA_IMG_withCellInput135K.npy', train_imgWCell)
        np.save(DATA + 'SAMPLETRAININGDATA_HEADING_withCellInput135K.npy', hotLabelHeading)
    if headingInput == True:
        train_imgWHeading = np.asarray(train_imgWHeading)
        hotLabelCell = np.asarray(hotLabelCell)
        np.save(DATA + 'SAMPLETRAININGDATA_IMG_withHeadingInput135K.npy', train_imgWHeading)
        np.save(DATA + 'SAMPLETRAININGDATA_CELL_withHeadingInput135K.npy', hotLabelCell)


    print('Done!')
    return train_imgWCell, hotLabelHeading, train_imgWHeading, hotLabelCell









def getLabels():
    #Places all labels of cells (that now have the correct images_per_cell) in one array. The random frames are placed
    #last in the array
    overLabels = cullOverRepped()
    underLabels, randLabels = addUnderRepped()
    allLabels = overLabels + underLabels + randLabels
    randStart = len(overLabels)+len(underLabels)
    np.save(DATA +'newdata_allFramesToBeProcessed135k.npy', allLabels)

    return allLabels, randStart





    ##############################################################################################
    ### Uncomment to preprocess data with randerasing and normalization (thru mean subtraction)###
    ### Don't forget to change resizeAndCrop() to convert to gray/alter the resizing method    ###
    ### (e.g. preserve aspect ratio vs squish  image)                                          ###
    ##############################################################################################

    # for i in range(len(training_data)):
    #     loading_bar(i,len(training_data))
    #     image = training_data[i][0]
    #     image = image - mean
    #     image = np.squeeze(image)
    #     re_image = randerase_image(image, 1)
    #     if re_image is None:
    #         re_image = image
    #
    #     training_data[i][0] = re_image

