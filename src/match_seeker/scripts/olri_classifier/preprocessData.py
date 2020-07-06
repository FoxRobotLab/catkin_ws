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


    def __init__(self, imageSize=100, imagesPerCell=500, numCells=271, numHeadings=8, imageDir="", dataFile=""):
        self.imageSize = imageSize
        self.imagesPerCell = imagesPerCell
        self.numCells = numCells
        self.numHeadings = numHeadings
        self.imageDir = imageDir
        self.dataFile = dataFile

        self.dataMean = None
        self.allFrames = []
        self.allImages = []
        self.allCellOutput = []
        self.allHeadingOutput = []
        self.cellsTooFewImages = []
        self.cellsEnoughImages = []
        self.frameData = {}
        self.cellData = {}
        self.headingData = {}
        self.buildDataDicts()


    def buildDataDicts(self, locBool=True, cell=True, heading=True):
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
            self.frameData[frameNum] = {}
            if locBool:
                self.frameData[frameNum]['loc'] = loc

            if cell:
                self.frameData[frameNum]['cell'] = cellNum

            if heading:
                self.frameData[frameNum]['heading'] = headingNum


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


    def generateTrainingData(self):
        """
        Fills in three instance variables with lists: self.allImages, which contains actual image files for the data,
        self.allCellOutput, which contains one-hot arrays corresponding to each image,
        self.allHeadingOutput, which contains one-hot arrays corresponding to each image
        :return: Nothing is returned, just setting instance variables
        """
        if len(self.frameData) == 0:
            print("Must read in data first")
            return

        # Create lists of frame numbers so that there are the right number frames per cell
        self.splitCellsByThreshold()
        enoughFrames = self.selectEnoughFrames()
        shortFrames, extraFrames = self.createMoreFrames()

        totalLen = len(enoughFrames) + len(shortFrames) + len(extraFrames)
        frameCount = 0
        for frame in (enoughFrames + shortFrames):
            print("Processing frame ", frameCount, "of", totalLen, "     (Frame number: ", frame,  ")")
            frameCount += 1
            self.processFrame(frame)


        for frame in extraFrames:
            print("Processing frame ", frameCount, "of", totalLen, "     (Frame number: ", frame, ")")
            frameCount += 1
            image = self.processFrame(frame, doRandErase=True)

            # training_data.append([np.array(image), self.makeOneHotList(int(frame_cell_dict[frame]), numCells),
            #                       self.makeOneHotList(int(frame_heading_dict[frame]) // 45, 8)])


        self.dataMean = self.calculateMean(self.allImages)

        # Note: removed some code here compared to the old olin_inputs_2019.py
        print('Done!')


    def processFrame(self, frameNum, doRandErase=False):
        """
        Given an image frame number, this does the actual preprocessing. It (1) resizes the image to be square,
        with the dimensions stored in this object, (2) converts the image to grayscale, and if doRandErase is True
        then it also performs a random erase on a rectangle of the image. It also builds one-hot arrays for
        both cell and heading, and then adds all the relevant data to the instance variable lists
        :param frameNum: The number of the frame to be read in
        :param doRandErase: Boolean, if True then an extra preprocessing step is performed: randErase
        :return: nothing
        """
        origImage = self.readImage(frameNum)
        if origImage is None:
            return
        resizedImage = cv2.resize(origImage, (self.imageSize, self.imageSize))
        grayResizedImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        if doRandErase:
            finalImage = self.randEraseImage(grayResizedImage)
        else:
            finalImage = grayResizedImage
        cellOneHot = self.makeOneHotList(self.frameData[frameNum]['cell'], self.numCells)
        headingIndex = self.frameData[frameNum]['heading'] // (360 // self.numHeadings)
        headOneHot = self.makeOneHotList(headingIndex, self.numHeadings)
        self.allFrames.append(frameNum)
        self.allImages.append(finalImage)
        self.allCellOutput.append(cellOneHot)
        self.allHeadingOutput.append(headOneHot)


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
                minInd = np.argmin(fListLens)
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


    def saveDataset(self, datasetFilename):
        """
        Saves the current allImages, allCellOutput, allHeadingOutput to a single file
        :param datasetFilename: Name of the file to save to
        :return:
        """
        if self.allImages == []:
            print("Must create dataset arrays first")
        else:
            np.savez(datasetFilename, images=self.allImages, frameNums=self.allFrames, cellOut=self.allCellOutput, headingOut=self.allHeadingOutput, mean=self.dataMean)



    def readImage(self, frameNum):
        """
        Takes in a frame number, and reads in and returns the image with that frame number.
        :param frameNum: The number of the frame
        :return: Returns an image array (or None if no image with that number)
        """
        fName = makeFilename(self.imageDir, frameNum)
        image = cv2.imread(fName)
        return image


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
        return reImage



def main():
    """
    Main program. Creates the preprocessor, from that generates the dataset, and then saves it to a file.
    :return: Nothing
    """
    preProc = DataPreprocess(imageDir=DATA + "frames/moreframes/",
                             dataFile=DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt")
    preProc.generateTrainingData()
    preProc.saveDataset(DATA + "susantestdataset")


if __name__ == "__main__":
    main()


