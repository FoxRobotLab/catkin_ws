"""--------------------------------------------------------------------------------
DataBalancing2022.py

Created: Summer 2022
Authors: Bea Bautista, Yifan Wu

This file reads in the master dictionary file of frames and can provide counts
of underrepresented and overrepresented cells or headings. The purpose of this
is for diagnostic purposes to guide data balancing/weighting of certain cells or headings.

--------------------------------------------------------------------------------"""


from frameCellMap import FrameCellMap
from paths import DATA
import numpy as np


class DataBalancer(object):
    def __init__(self):
        self.dictFile = DATA + "frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt"
        self.labelMap = FrameCellMap(self.dictFile)
        self.cellCountsMap = {}
        self.headingCountsMap = {}


    def getCellCounts(self):
        cellCounts = {}
        for cell in self.labelMap.cellData.keys():
            frameCount = 0
            for i in range(len(self.labelMap.cellData.get(cell))):
                frameCount += 1
            cellCounts[cell] = frameCount
        self.cellCountsMap = cellCounts
        return cellCounts


    def getHeadingCounts(self):
        headingCounts = {}
        for heading in self.labelMap.headingData.keys():
            frameCount = 0
            for i in range(len(self.labelMap.headingData.get(heading))):
                frameCount += 1
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



if __name__ == '__main__':
    balancer = DataBalancer()
    print(balancer.getCellCounts())
    print(balancer.getHeadingCounts())
    print(balancer.getTotalCount())
    print(balancer.getUnderRepCells(500))
    print(balancer.getUnderRepHeadings(10000))
    print(balancer.getOverRepCells(500))
    print(balancer.getOverRepHeadings(10000))

