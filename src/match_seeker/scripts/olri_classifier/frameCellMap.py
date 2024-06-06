"""
File: frameCellMap.py
Created: June 2020
Updated: June 2022
Authors: Susan Fox

Because the generator version has been created, this file was modified so that it only reads in the information
mapping cells to (x, y) coordinatrs and associating those locations with frames. The rest is commented out for now
and will be removed once we verify that it is no longer needed.
"""


import cv2
import numpy as np
import os
import random
from datetime import datetime

#import paths from Susan's Machine
# from paths import DATA, DATA2022, pathToMatchSeeker

#import paths from Precision 5820
from paths import DATA, DATA2022

from imageFileUtils import makeFilename

# from DataPaths import cellMapData, basePath
from paths import DATA


class FrameCellMap(object):

    def __init__(self, dataFile, cellFile, format="old"):
        self.dataFile = dataFile
        self.cellFile = cellFile
        self.format = format
        self.allFrames = []
        self.allImages = []
        self.allCellOutput = []
        self.allHeadingOutput = []
        self.cellsTooFewImages = []
        self.cellsEnoughImages = []
        self.allxyh = []
        self.frameData = {}
        self.cellData = {}
        self.locData = {}
        self.headingData = {}
        self.readCells()
        if format == "old":
            self.buildDataDictsOlder(locBool=False)
        else:
            self.buildDataDictsNewer(locBool=False)

        self.dumbNum = 0
        # TODO: Figure out what the badLocDict is all about
        self.badLocDict = {140: (30, 57), 141: (32, 57), 185: (10, 89), 186: (10, 87), 187: (10, 85), 188: (10, 83),
                           189: (10, 81), 190: (10, 79), 215: (6, 85), 216: (6, 87), 217: (6, 89)}


    def readCells(self):
        """Reads in cell data, building a dictionary to hold it."""
        cellDict = dict()
        try:
            with open(self.cellFile, 'r') as cellF:
                for line in cellF:
                    if line[0] == '#' or line.isspace():
                        continue
                    parts = line.split()
                    cellNum = parts[0]
                    locList = [float(v) for v in parts[1:]]
                    # print("Cell " + cellNum + ": ", locList)
                    cellDict[cellNum] = locList
        except:
            print("Error reading cell data file:", self.cellData)
        self.cellBorders = cellDict


    def buildDataDictsOlder(self, locBool=True, cell=True, heading=True):
        """
        Reads in the data in the self.dataFile file, and fills in various dictionaries.
        self.frameData uses the frames number as the key and contains a dictionary with keys 'cell', 'heading', 'loc'
        self.cellFile uses the cell number as the key, and the frames number as the value
        self.headingData uses the heading number as the key, and the frames number as the value
        :return: nothing
        """
        try:
            with open(self.dataFile) as frameData:
                for line in frameData:
                    splitList = line.split()
                    frameNum = int(splitList[0])
                    cellNum = int(splitList[1])
                    xVal = float(splitList[2])
                    yVal = float(splitList[3])
                    headingNum = int(splitList[4])
                    loc = (xVal, yVal)
                    self.frameData[frameNum] = {'loc': loc, 'cell': cellNum, 'heading': headingNum, 'frameNum': frameNum}

                    if cellNum not in self.cellData:
                        self.cellData[cellNum] = {frameNum}
                    else:
                        self.cellData[cellNum].add(frameNum)

                    if headingNum not in self.headingData:
                        self.headingData[headingNum] = {frameNum}
                    else:
                        self.headingData[headingNum].add(frameNum)

                    if loc not in self.locData:
                        self.locData[loc] = {frameNum}
                    else:
                        self.locData[loc].add(frameNum)

            if locBool:
                for frame in self.frameData:
                    loc = self.frameData[frame]['loc']
                    cell = self.frameData[frame]['cell']
                    calcCell = self.convertLocToCell(loc)
                    # If (x, y) coordinate doesn't match the assigned cell from the data file, then change the (x, y)
                    # to be the one associated with this cell in the badLocDict dictionary
                    if int(calcCell) != int(cell):
                        # print(calcCell, cell)
                        self.frameData[frame]['loc'] = self.badLocDict[cell]
        except:
            print("Failed to open and read dictionary file:", self.dataFile)

    def buildDataDictsNewer(self, locBool=True, cell=True, heading=True):
        """
        Reads in the data in the self.dataFile file, and fills in various dictionaries.
        self.frameData uses the frames number as the key and contains a dictionary with keys 'cell', 'heading', 'loc'
        self.cellFile uses the cell number as the key, and the frames number as the value
        self.headingData uses the heading number as the key, and the frames number as the value
        :return: nothing
        """
        try:
            with open(self.dataFile) as frameData:
                for line in frameData:
                    splitList = line.split()
                    frameName = splitList[0]
                    xVal = float(splitList[1])
                    yVal = float(splitList[2])
                    cellNum = int(splitList[3])
                    headingNum = int(splitList[4])
                    timeStamp = splitList[5]
                    loc = (xVal, yVal)
                    self.frameData[frameName] = {'loc': loc, 'cell': cellNum, 'heading': headingNum, 'frameName': frameName}

                    if cellNum not in self.cellData:
                        self.cellData[cellNum] = {frameName}
                    else:
                        self.cellData[cellNum].add(frameName)

                    if headingNum not in self.headingData:
                        self.headingData[headingNum] = {frameName}
                    else:
                        self.headingData[headingNum].add(frameName)

                    if loc not in self.locData:
                        self.locData[loc] = {frameName}
                    else:
                        self.locData[loc].add(frameName)

            if locBool:
                for frame in self.frameData:
                    loc = self.frameData[frame]['loc']
                    cell = self.frameData[frame]['cell']
                    calcCell = self.convertLocToCell(loc)
                    # If (x, y) coordinate doesn't match the assigned cell from the data file, then change the (x, y)
                    # to be the one associated with this cell in the badLocDict dictionary
                    if int(calcCell) != int(cell):
                        # print(calcCell, cell)
                        self.frameData[frame]['loc'] = self.badLocDict[cell]
        except:
            print("Failed to open and read dictionary file:", self.dataFile)


    # def buildDataDictsOneRun(self):
    #     """
    #     Modified version of buildDataDicts that reads in the data in the self.dataFile file, and fills in
    #     various dictionaries. This function is for reading in new FrameData text files from individual runs
    #     on Cutie.
    #
    #     self.frameData uses the frames name as the key and contains a dictionary with keys 'cell', 'heading', 'timestamp', 'loc', 'xval', 'yval'
    #     self.cellFile uses the cell number as the key, and a list of frames names as the value
    #     self.headingData uses the heading number as the key, and a list of the frames names as the value
    #     :return: nothing
    #     """
    #
    #     with open(self.dataFile) as frameData:
    #         for line in frameData:
    #             splitList = line.split()
    #             frameName = splitList[0]
    #             xVal = float(splitList[1])
    #             yVal = float(splitList[2])
    #             cellNum = int(splitList[3])
    #             headingNum = int(splitList[4])
    #             timeStamp = splitList[5]
    #             loc = (xVal, yVal)
    #             self.frameData[frameName] = {'cell': cellNum, 'heading': headingNum, 'timestamp': timeStamp, 'loc': loc, 'xval': xVal, 'yval': yVal }
    #
    #             if cellNum not in self.cellData:
    #                 self.cellData[cellNum] = {frameName}
    #             else:
    #                 self.cellData[cellNum].add(frameName)
    #
    #             if headingNum not in self.headingData:
    #                 self.headingData[headingNum] = {frameName}
    #             else:
    #                 self.headingData[headingNum].add(frameName)
    #
    #             if loc not in self.locData:
    #                 self.locData[loc] = {frameName}
    #             else:
    #                 self.locData[loc].add(frameName)


    # def generateTrainingData(self):
    #     """
    #     Fills in three instance variables with lists: self.allImages, which contains actual image files for the data,
    #     self.allCellOutput, which contains one-hot arrays corresponding to each image,
    #     self.allHeadingOutput, which contains one-hot arrays corresponding to each image
    #     :return: Nothing is returned, just setting instance variables
    #     """
    #     if len(self.frameData) == 0:
    #         print("Must read in data first")
    #         return
    #
    #     # Create lists of frames numbers so that there are the right number frames per cell
    #     self.splitCellsByThreshold()
    #     enoughFrames = self.selectEnoughFrames()
    #     shortFrames, extraFrames = self.createMoreFrames()
    #
    #     totalLen = len(enoughFrames) + len(shortFrames) + len(extraFrames)
    #     frameCount = 0
    #     for frame in (enoughFrames + shortFrames):
    #         print("Processing frames ", frameCount, "of", totalLen, "     (Frame number: ", frame,  ")")
    #         frameCount += 1
    #         self.processFrame(frame)
    #
    #
    #     for frame in extraFrames:
    #         print("Processing frames ", frameCount, "of", totalLen, "     (Frame number: ", frame, ")")
    #         frameCount += 1
    #         image = self.processFrame(frame, doRandErase=True)
    #
    #         # training_data.append([np.array(image), self.makeOneHotList(int(frame_cell_dict[frames]), numCells),
    #         #                       self.makeOneHotList(int(frame_heading_dict[frames]) // 45, 8)])
    #
    #
    #     self.dataMean = self.calculateMean(self.allImages)
    #
    #     # Note: removed some code here compared to the old olin_inputs_2019.py
    #     print('Done!')


    # def processFrame(self, frameNum, doRandErase=False):
    #     """
    #     Given an image frames number, this does the actual preprocessing. It (1) resizes the image to be square,
    #     with the dimensions stored in this object, (2) converts the image to grayscale, and if doRandErase is True
    #     then it also performs a random erase on a rectangle of the image. It also builds one-hot arrays for
    #     both cell and heading, and then adds all the relevant data to the instance variable lists
    #     :param frameNum: The number of the frames to be read in
    #     :param doRandErase: Boolean, if True then an extra preprocessing step is performed: randErase
    #     :return: nothing
    #     """
    #     origImage = self.readImage(frameNum)
    #
    #     if origImage is None:
    #         return
    #     resizedImage = cv2.resize(origImage, (self.imageSize, self.imageSize))
    #     grayResizedImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
    #     if doRandErase:
    #         finalImage = self.randEraseImage(grayResizedImage)
    #     else:
    #         finalImage = grayResizedImage
    #
    #     cv2.imshow("Original", origImage)
    #     cv2.imshow("Resized", resizedImage)
    #     cv2.imshow("grayResized", grayResizedImage)
    #     cv2.imshow("RandErased", finalImage)
    #     cv2.waitKey()
    #
    #     cellOneHot = self.makeOneHotList(self.frameData[frameNum]['cell'], self.numCells)
    #     headingIndex = self.frameData[frameNum]['heading'] // (360 // self.numHeadings)
    #     headOneHot = self.makeOneHotList(headingIndex, self.numHeadings)
    #
    #     x,y = self.frameData[frameNum]['loc']
    #     xyh = []
    #     xyh.append(x)
    #     xyh.append(y)
    #     xyh.append(self.frameData[frameNum]['heading'])
    #
    #     self.allFrames.append(frameNum)
    #     self.allImages.append(finalImage)
    #     self.allCellOutput.append(cellOneHot)
    #     self.allHeadingOutput.append(headOneHot)
    #     self.allxyh.append(xyh)



    # def splitCellsByThreshold(self):
    #     """
    #     Determines which cells are below the threshold for "enough" frames per cell, versus those who are sufficiently or
    #     "over" represented. Saves the values in instance variables
    #     :return: Nothing
    #     """
    #     tooFew = []
    #     enough = []
    #
    #     for cell in self.cellFile:
    #         numFrames = len(self.cellFile[cell])
    #         if numFrames <= self.imagesPerCell:
    #             tooFew.append(cell)
    #         else:
    #             enough.append(cell)
    #     self.cellsTooFewImages = tooFew
    #     self.cellsEnoughImages = enough


    # def selectEnoughFrames(self):
    #     """
    #     Looks at each cell that has enough frames, and chooses frames for each heading, choosing from each heading in
    #     turn until headings run out of frames.
    #     :return: list of frames selected so that we have self.imagesPerCell frames for each cell (for all cells that
    #     have at least self.imagesPerCell frames recorded.
    #     """
    #     chosenFrames = []
    #
    #     for cell in self.cellsEnoughImages:
    #         framesForCell = []
    #         framesEachHeading = {}
    #
    #         # Build a list for each heading of all frames for this cell that are for this heading
    #         for fr in self.cellFile[cell]:
    #             frHead = self.frameData[fr]['heading']
    #             if frHead in framesEachHeading:
    #                 framesEachHeading[frHead].append(fr)
    #             else:
    #                 framesEachHeading[frHead] = [fr]
    #         frameLists = list(framesEachHeading.values())
    #         fListLens = [len(l) for l in frameLists]
    #         listInd = 0
    #
    #         # loops through each heading index, and randomly selects a frames from that heading list, skipping []
    #         while len(framesForCell) < self.imagesPerCell:
    #             if len(frameLists[listInd]) > 0:
    #                 pickedFrame = random.choice(frameLists[listInd])
    #                 frameLists[listInd].remove(pickedFrame)
    #                 fListLens[listInd] -= 1
    #                 framesForCell.append(pickedFrame)
    #             listInd = (listInd + 1) % len(frameLists)
    #         chosenFrames += framesForCell
    #     return chosenFrames

    def selectNFramesAllCells(self, n):
        """
        Simplified version of selectEnoughFrames which randomly selects n number of images per cell
        :param n: number of frames to be selected for each cell
        :return chosenFrames: map of frames per cell, with cell numbers for keys and list of frames as values
        """
        chosenFrames = {}

        # missingCells = []
        # for i in range(271):
        #     if i not in self.cellFile.keys():
        #         missingCells.append(i)
        # print('MISSING', missingCells)
        # print('--------')

        for cell in self.cellData.keys():
            framesForCell = self.selectNFramesOneCell(cell, n)
            chosenFrames[cell] = framesForCell
        return chosenFrames

    def selectNFramesOneCell(self, cell, n):
        """
        Helper function for testing to select n frames for one specific cell.
        If there are less frames than the desired number of frames n, function will randomly
        select n images of that cell with repetition
        :param cell: cell number to select frames for
        :param n: number of photos to randomly select per cell
        :return framesForCell: list of randomly selected frame numbers for the inputted cell
        """
        framesForCell = []
        cellFrames = self.cellData[cell]
        while len(framesForCell) < n:
            if len(cellFrames) < n:
                print("There are less frames than ", n, ". There are ", len(cellFrames), " frames for cell ", cell)
                randImage = random.choice(list(cellFrames))
                framesForCell.append(randImage)
            else:
                randImage = random.choice(list(cellFrames))
                if randImage not in framesForCell:
                    framesForCell.append(randImage)
        return framesForCell


    def selectNFramesAllHeadings(self, n):
        """
        Randomly selects n number of frames for each heading.
        :param n: number of frames to be selected for each heading
        :return chosenFrames: map of frames per heading, with headings for keys and list of frames as values
        """
        chosenFrames = {}
        for heading in self.headingData.keys():
            framesForHeading = self.selectNFramesOneHeading(heading, n)
            chosenFrames[heading] = framesForHeading
        return chosenFrames


    def selectNFramesOneHeading(self, heading, n):
        """
        Helper function for testing to select n frames for one specific heading.
        If there are less frames than the desired number of frames n, function will randomly
        select n images of that heading with repetition
        :param heading: heading number to select frames for
        :param n: number of photos to randomly select per heading
        :return framesForHeading: list of randomly selected frame numbers for the inputted heading
        """
        framesForHeading = []
        while len(framesForHeading) < n:
            # if there are fewer images than n, pick with repetition
            if len(self.headingData[heading]) < n:
                randImage = random.choice(list(self.headingData[heading]))
                framesForHeading.append(randImage)
            else:
                # do not pick with repetition
                randImage = random.choice(list(self.headingData[heading]))
                if randImage not in framesForHeading:
                    framesForHeading.append(randImage)
        return framesForHeading

    # def createMoreFrames(self):
    #     """
    #     Looks at each cell that has enough frames, and chooses frames for each heading, choosing from each heading in
    #     turn until headings run out of frames.
    #     :return: list of frames selected so that we have self.imagesPerCell frames for each cell (for all cells that
    #     have at least self.imagesPerCell frames recorded.
    #     """
    #     chosenFrames = []
    #     newFrames = []
    #
    #     for cell in self.cellsTooFewImages:
    #         oldFramesForCell = []
    #         newFramesForCell = []
    #         framesEachHeading = {}
    #
    #         # Build a list for each heading of all frames for this cell that are for this heading
    #         for fr in self.cellFile[cell]:
    #             oldFramesForCell.append(fr)
    #             frHead = self.frameData[fr]['heading']
    #             if frHead in framesEachHeading:
    #                 framesEachHeading[frHead].append(fr)
    #             else:
    #                 framesEachHeading[frHead] = [fr]
    #         frameLists = list(framesEachHeading.values())
    #         fListLens = [len(l) for l in frameLists]
    #
    #         # loops through each heading index, and randomly selects a frames from that heading list, skipping []
    #         while len(oldFramesForCell) + len(newFramesForCell) < self.imagesPerCell:
    #             minInd = np.argmin(fListLens)
    #             if len(frameLists[minInd]) > 0:
    #                 pickedFrame = random.choice(frameLists[minInd])
    #                 fListLens[minInd] += 1
    #                 newFramesForCell.append(pickedFrame)
    #         chosenFrames += oldFramesForCell
    #         newFrames += newFramesForCell
    #     return chosenFrames, newFrames
    #
    #
    # def makeOneHotList(self, index, size):
    #     """
    #     Constructs a "one-hot" list that is all zeros, except for the position given by the index, which is one.
    #     :param index: the position in the one-hot list that should be one
    #     :param size:  the length of the one-hot list
    #     :return: a list of length size, all zeros, except for a 1 at position index
    #     """
    #     onehot = [0] * size
    #     onehot[index] = 1
    #     return onehot


    # def calculateMean(self, images):
    #     """
    #     Takes a set of images, and computes a pixel-wise average brightness or color. The resulting matrix is returned
    #     :param images: A list of images, could be grayscale, color, or grayscale with a second channel added
    #     :return: a mean image the same shape as one of the list of images, pixel-wise average brightness or color
    #     """
    #     dimens = images[0].shape
    #
    #     if (len(dimens) == 2) or ((len(dimens) == 3) and (dimens[2] == 3)):  # this is a grayscale image or a color image
    #         N = 0
    #         mean = np.zeros(dimens, np.float64)
    #         for img in images:
    #             mean += img
    #             N += 1
    #         mean /= N
    #         return mean
    #     elif (len(dimens) == 3) and (dimens[2] == 2):  # this is a gray image combined with an extra channel
    #         N = 0
    #         mean = np.zeros((dimens[0], dimens[1]), np.float64)
    #         for img in images:
    #             mean += img[:,:,0]
    #             N += 1
    #         mean /= N
    #         return mean
    #     else:
    #         return 0.0


    # def saveDataset(self, datasetFilename):
    #     """
    #     Saves the current allImages, allCellOutput, allHeadingOutput to a single file
    #     :param datasetFilename: Name of the file to save to
    #     :return:
    #     """
    #     if self.allImages == []:
    #         print("Must create dataset arrays first")
    #     else:
    #         np.savez(datasetFilename, images=self.allImages, frameNums=self.allFrames, xyhOut=self.allxyh, cellOut=self.allCellOutput, headingOut=self.allHeadingOutput, mean=self.dataMean)



    # def readImage(self, frameNum):
    #     """
    #     Takes in a frames number, and reads in and returns the image with that frames number.
    #     :param frameNum: The number of the frames
    #     :return: Returns an image array (or None if no image with that number)
    #     """
    #     fName = makeFilename(self.imageDir, frameNum)
    #     image = cv2.imread(fName)
    #     return image


    # def randEraseImage(self, image, minSize=0.02, maxSize=0.4, minRatio=0.3, maxRatio=1 / 0.3, minVal=0, maxVal=255):
    #     """
    #     Randomly erase a rectangular part of the given image in order to augment data. Based on code from github:
    #     https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
    #     :param image: The image to be modified
    #     :param minSize: The smallest size of the rectangle to be erased, in pixels
    #     :param maxSize: The largest size of the rectangle to be erased, in pixels
    #     :param minRatio: Minimum scaling factor used for the width versus height of rectangle to erase
    #     :param maxRatio: Maximum scaling factor used for the width versus height of rectangle to erase
    #     :param minVal: minimum brightness value to set erased rectangle to
    #     :param maxVal: maximum brightness value to set erased rectangle to
    #     :return: Returns a new image, with a randomly placed and sized rectangle of a random color placed on it
    #     """
    #     global lc, tr, rc, br
    #     assert len(image.shape) == 2
    #
    #     if self.dumbNum == 0:
    #         lc = 0
    #         tr = 0
    #         rc = 0
    #         br = 0
    #         self.dumbNum = 1
    #
    #
    #     reImage = image.copy()
    #     brightness = np.random.uniform(minVal, maxVal)
    #     h, w = reImage.shape
    #
    #     size = np.random.uniform(minSize, maxSize) * h * w
    #     ratio = np.random.uniform(minRatio, maxRatio)
    #     width = int(np.sqrt(size / ratio))
    #     height = int(np.sqrt(size * ratio))
    #
    #     midRow = np.random.randint(0, h)
    #     midCol = np.random.randint(0, w)
    #     topRow = midRow - (height // 2)
    #     leftCol = midCol - (width // 2)
    #     reImage[topRow:topRow + height, leftCol:leftCol + width] = brightness
    #     if topRow < 0:
    #         tr += 1
    #     if leftCol < 0:
    #         lc += 1
    #     if (topRow + height) >= h:
    #         br += 1
    #     if (leftCol + width) >= w:
    #         rc += 1
    #     return reImage


    def convertLocToCell(self, pose):
        """Takes in a location that has 2 or 3 values and reports the cell, if any, that it is a part
        of."""
        x = pose[0]
        y = pose[1]

        for cell in self.cellBorders:
            [x1, y1, x2, y2] = self.cellBorders[cell]
            if (x1 <= x < x2) and (y1 <= y < y2):
                return cell

        return -1 #TODO: It should not think it is outside the map




def main():
    """
    Main program. Creates the preprocessor, from that generates the dataset, and then saves it to a file.
    :return: Nothing
    """
    print("FrameCellMap loading")
    # dataMap = FrameCellMap(dataFile=DATA + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt",
    #                        cellFile=pathToMatchSeeker + "res/map/mapToCells.txt")
                            # imageDir=DATA + "moreframes/",
    dataPath = DATA2022 + "FrameDataReviewed-20220708-11:06frames.txt"
    print("  dataPath:", dataPath)
    cellPath = pathToMatchSeeker + "res/map/mapToCells.txt"
    print("   cellPath:", cellPath)
    dataMap = FrameCellMap(dataFile=dataPath, cellFile=cellPath, format="new")

print("Done loading")

    # preProc.generateTrainingData()
    # preProc.saveDataset(DATA + "regressionTestSet")
    # print("Processing frames")
    # preProc.processFrame(4954, True)

if __name__ == "__main__":
    main()






