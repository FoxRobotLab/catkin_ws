"""--------------------------------------------------------------------------------
DataBalancing2022.py

Created: Summer 2022
Authors: Bea Bautista, Yifan Wu

This file reads in the master dictionary file of the old dataset of 95k frames and can provide
counts of underrepresented and overrepresented cells or headings. The purpose of this
is for diagnostic purposes to guide data balancing/weighting of certain cells or headings.

Updated in July 2022 to read in text files of ongoing data collection

--------------------------------------------------------------------------------"""
import re

from frameCellMap import FrameCellMap
from paths import DATA, data2022
import numpy as np
import math
import time


class DataBalancer(object):
    def __init__(self, dictFileName = None, mergeFrameCountFile = None):
        self.dictFileName = dictFileName
        self.dictFile = data2022 + dictFileName
        self.mergeFrameCountFile = data2022 + mergeFrameCountFile
        self.headingData = {}
        self.cellData = {}

        if dictFileName == None or mergeFrameCountFile == None:
            self.oldDataset95k = DATA + "MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
            self.labelMap = FrameCellMap(self.oldDataset95k)

            self.cellCountsMap = self._countCells(self.labelMap.cellData)
            self.headingCountsMap = self._countHeadings(self.labelMap.headingData)
        else:
            self._readNewDict()
            self.cellCountsMap = self._countCells(self.cellData)
            self.headingCountsMap = self._countHeadings(self.headingData)


    def _readNewDict(self):
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
        for i, line in enumerate(prevFile):
            if "!" in line:
                continue
            if "*" in line:
                splitList = line.split()
                prevTotalFrames = int(splitList[2])
            #Save cell counts
            if i > 2 and i < 274:
                splitList = line.split()
                cellNum = int(splitList[0])
                frameCount = int(splitList[1])
                prevCellCounts[cellNum] = frameCount
            #Save heading counts
            elif i >= 274:
                splitList = line.split()
                headingNum = int(splitList[0])
                frameCount = int(splitList[1])
                prevHeadingCounts[headingNum] = frameCount

        srcDictTimeStamp = re.sub('[a-zA-Z]', '', self.dictFileName)

        logName = "NewFrameCountMerged" + srcDictTimeStamp + "txt"
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        numCells = 271
        try:
            logFile = open(data2022 + logName, 'w')
        except:
            print("FAILED TO OPEN DATA FILE")

        logFile.write("! Counts from " + self.dictFileName + " merged with " + self.mergeFrameCountFile + "\n")
        logFile.write("* NumFrames " + str(self.getTotalCount() + prevTotalFrames) + "\n")

        logFile.write("! CELL COUNTS \n" )
        for i in range(numCells):
            logFile.write(str(i) + " " + str(self.cellCountsMap.get(i, 0) + prevCellCounts.get(i, 0)) + "\n")

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

    balancerNew = DataBalancer(dictFileName="FrameData-20220705-16:16frames.txt", mergeFrameCountFile="NewFrameCount-20220706-15:18.txt")
    print(balancerNew.cellCountsMap)
    print(balancerNew.headingCountsMap)
    balancerNew.mergeCounts()
