"""--------------------------------------------------------------------------------
DataBalancing2022.py

Created: June 2022
Authors: Bea Bautista, Yifan Wu
Updated July 2022

DataBalancer counts the number of frames per cell and number of frames per heading by taking in a
text file of raw frame information, with the option to merge the counts from another frame count text
file passed in as mergeFrameCountFile. DataBalancer can save the frame counts or the merged frame counts
into a new text file inside classifier2022data.

If a text file is not passed into the constructor, DataBalancer by default uses the previous master file
of the old 95k frame dataset located inside the classifier2019data folder.

DataBalancer also has methods for returning a dictionary of weights per cell or heading if the data is unbalanced.
These weight dictionaries can be passed into the 2022 RGB heading or cell classification models inside the train method.

The purpose of this class is for diagnostic purposes to guide data balancing/weighting of certain cells or headings,
and data collection in 2022 to ensure that we collect enough frames per heading or cell.

--------------------------------------------------------------------------------"""

from frameCellMap import FrameCellMap
from paths import DATA, DATA2022, pathToMatchSeeker
# from paths import DATA, data2022
import numpy as np
import math
import re
import time


class DataBalancer(object):
    def __init__(self, dictFileName = None, mergeFrameCountFile = None):
        #File names and file paths for the text files to be read and counted, and text file for counts to merge
        self.dictFileName = dictFileName
        self.dictFile = data2022 + str(dictFileName)
        self.mergeFrameCountFileName = mergeFrameCountFile
        self.mergeFrameCountFile = data2022 + str(mergeFrameCountFile)

        #Dictionaries for holding cell and heading counts from self.dictFileName
        self.headingData = {}
        self.cellData = {}

        #If no text files are passed in, count the old 95k dataset by default
        if dictFileName == None and mergeFrameCountFile == None:
            self.oldDataset95k = DATA + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
            self.labelMap = FrameCellMap(dataFile=self.oldDataset95k, cellFile = pathToMatchSeeker + "res/map/mapToCells.txt")

            self.cellCountsMap = self._countCells(self.labelMap.cellData)
            self.headingCountsMap = self._countHeadings(self.labelMap.headingData)
        else:
            self._readNewDict()
            self.cellCountsMap = self._countCells(self.cellData)
            self.headingCountsMap = self._countHeadings(self.headingData)


    def _readNewDict(self):
        """
        Reads the raw frame information file (self.dictFile) and initializes
        self.cellData and self.headingData which hold cell/heading numbers as keys
        and a list of frames as values.
        """
        try:
            with open(self.dictFile) as frameData:
                for line in frameData:
                    splitList = line.split()
                    frameName = splitList[0]
                    xVal = float(splitList[1])
                    yVal = float(splitList[2])
                    cellNum = int(splitList[3])
                    headingNum = int(splitList[4])
                    timeStamp = splitList[5]
                    if cellNum not in self.cellData:
                        self.cellData[cellNum] = {frameName}
                    else:
                        self.cellData[cellNum].add(frameName)

                    if headingNum not in self.headingData:
                        self.headingData[headingNum] = {frameName}
                    else:
                        self.headingData[headingNum].add(frameName)
        except:
            print("Failed to open dictionary File")




    def _countCells(self, map):
        cellCounts = {}
        for cell in map.keys():
            frameCount = len(map.get(cell))
            cellCounts[cell] = frameCount
        self.cellCountsMap = cellCounts
        return cellCounts


    def _countHeadings(self, map):
        headingCounts = {}
        for heading in map.keys():
            frameCount = len(map.get(heading))
            headingCounts[heading] = frameCount
        self.headingCountsMap = headingCounts
        return headingCounts


    def getTotalCount(self):
        totalFramesCell = 0
        totalFramesHeading = 0
        for cell in self.cellCountsMap:
            totalFramesCell += self.cellCountsMap.get(cell,0)
        for heading in self.headingCountsMap:
            totalFramesHeading += self.headingCountsMap.get(heading,0)
        assert totalFramesHeading == totalFramesCell
        return totalFramesCell


    def writeCurrentCounts(self):
        """
        Saves the frame counts per cell and frame count per heading to a new text file inside classifier2022data
        The saved text file is meant to keep track of the progress per cell/heading made for new data collection.
        """
        srcDictTimeStamp = re.sub('[a-zA-Z]', '', self.dictFileName)

        logName = "NewFrameCount" + srcDictTimeStamp + ".txt"
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        numCells = 271
        try:
            logFile = open(data2022 + logName, 'w')
        except:
            print("FAILED TO OPEN DATA FILE")

        logFile.write("! Counts from " + self.dictFileName + "\n")
        logFile.write("* NumFrames " + str(self.getTotalCount()) + "\n")

        logFile.write("! CELL COUNTS \n" )
        for i in range(numCells):
            logFile.write(str(i) + " " + str(self.cellCountsMap.get(i, 0)) + "\n")

        logFile.write("! HEADING COUNTS \n")
        for heading in potentialHeadings:
            logFile.write(str(heading) + " " + str(self.headingCountsMap.get(heading, 0)) + "\n")

        logFile.close()


    def mergeCounts(self):
        """
        Uses the older count file passed into the constructor as self.mergeFrameCountFile and the newer
        raw frame data passed in as self.dictFileName to save a new merged cell/heading frame count text file
        from multiple data collection runs.
        """
        if self.mergeFrameCountFile == None:
            print("Frame Count Text File not Found. Cannot merge into new file.")
            return
        try:
            prevFile = open(self.mergeFrameCountFile, 'r')
        except:
            print("FAILED TO OPEN PREV DATA FILE FOR MERGE")

        prevCellCounts = {}
        prevHeadingCounts = {}
        prevTotalFrames = 0

        sourceFiles = ""
        cellStartLine, headingStartLine = 1e8, 1e8

        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        numCells = 271

        for i, line in enumerate(prevFile):
            if i == 0 or "," in line:
                sourceFiles += line
            if "! CELL COUNTS" in line:
                cellStartLine = i
            elif "! HEADING COUNTS" in line:
                headingStartLine = i
            if "*" in line:
                splitList = line.split()
                prevTotalFrames = int(splitList[2])
            #Save cell counts
            if i > cellStartLine and i < (cellStartLine + numCells -1):
                splitList = line.split()
                cellNum = int(splitList[0])
                frameCount = int(splitList[1])
                prevCellCounts[cellNum] = frameCount
            #Save heading counts
            elif i > headingStartLine:
                splitList = line.split()
                headingNum = int(splitList[0])
                frameCount = int(splitList[1])
                prevHeadingCounts[headingNum] = frameCount

        srcDictTimeStamp = re.sub('[a-zA-Z]', '', self.dictFileName)
        logName = "NewFrameCountMerged" + srcDictTimeStamp + "txt"

        try:
            logFile = open(data2022 + logName, 'w')
        except:
            print("FAILED TO OPEN DATA FILE")

        logFile.write(str(sourceFiles) + "\n")
        logFile.write("Latest file merged: " + self.dictFileName + "\n")
        logFile.write("* CumulativeNumFrames " + str(self.getTotalCount() + prevTotalFrames) + "\n \n")

        logFile.write("! CELL COUNTS \n" )
        for i in range(numCells):
            logFile.write(str(i) + " " + str(self.cellCountsMap.get(i, 0) + prevCellCounts.get(i, 0)) + "\n")

        logFile.write("\n")
        logFile.write("! HEADING COUNTS \n")
        for heading in potentialHeadings:
            logFile.write(str(heading) + " " + str(self.headingCountsMap.get(heading, 0) + prevHeadingCounts.get(heading, 0)) + "\n")

        logFile.close()


    def getUnderRepCells(self, nImages):
        underRepCells = {}
        for cell in self.cellCountsMap:
            if self.cellCountsMap.get(cell) < nImages:
                underRepCells[cell] = self.cellCountsMap[cell]
        return underRepCells


    def getUnderRepHeadings(self, nImages):
        underRepHeadings = {}
        for heading in self.headingCountsMap:
            if self.headingCountsMap.get(heading) < nImages:
                underRepHeadings[heading] = self.headingCountsMap[heading]
        return underRepHeadings

    def getOverRepCells(self, nImages):
        underRepCells = {}
        for cell in self.cellCountsMap:
            if self.cellCountsMap.get(cell) >= nImages:
                underRepCells[cell] = self.cellCountsMap[cell]
        return underRepCells


    def getOverRepHeadings(self, nImages):
        underRepHeadings = {}
        for heading in self.headingCountsMap:
            if self.headingCountsMap.get(heading) >= nImages:
                underRepHeadings[heading] = self.headingCountsMap[heading]
        return underRepHeadings


    def getClassWeightCells(self, mu=0.15):
        total = np.sum(list(self.cellCountsMap.values()))
        keys = self.cellCountsMap.keys()
        class_weight = dict()
        for i in range(271):
            if i in keys:
                score = math.log(mu * total / float(self.cellCountsMap.get(i, 0)))
                class_weight[i] = score if score > 1.0 else 1.0
            else:
                class_weight[i] = 0
        return class_weight

    def getClassWeightHeadings(self, mu=0.15):
        total = np.sum(list(self.headingCountsMap.values()))
        keys = self.headingCountsMap.keys()
        class_weight = dict()
        for key in keys:
            score = math.log(mu*total  / float(self.headingCountsMap.get(key, 0)))
            class_weight[key] = score if score > 1.0 else 1.0
        return class_weight



if __name__ == '__main__':
    # balancer = DataBalancer()
    # print(balancer.cellCountsMap)
    # print(balancer.headingCountsMap)

    balancerNew = DataBalancer(dictFileName="FrameDataReviewed-20220708-15:36frames", mergeFrameCountFile="FrameCountMerged-20220708-11:06txt")
    balancerNew.mergeCounts()
    # print(balancerNew.cellCountsMap)
    # print(balancerNew.headingCountsMap)
