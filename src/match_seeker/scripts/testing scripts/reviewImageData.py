"""--------------------------------------------------------------------------------------------------------------------
This program was built off of the walkAndTimestamp2022 program by Malini Sharma. It allows a user to view simultaneously
video frames and data files to write a new text file that correctly assigns the timestamp, coordinates, heading, and
cell number to a given image.

Authors: Oscar Reza and Elisa Avalos
--------------------------------------------------------------------------------------------------------------------"""

import os
import time

import cv2
import numpy as np

from src.match_seeker.scripts.olri_classifier.OlinWorldMap import WorldMap
from src.match_seeker.scripts.olri_classifier.DataPaths import basePath


class ImageReview(object):

    def __init__(self, output_file_path):
        """Set up data to be held."""
        self.currLoc = (0, 0)
        self.currHeading = 0

        # variables for managing data source
        self.clickNum = 1
        self.outputFilePath = output_file_path

        # Olin Map
        self.olinMap = WorldMap()

        # instance variables to hold displayed images
        self.locGui = None
        self.origMap = None
        self.currMap = None
        self.currFrameIndex = 0
        self.fileList = self._getDataFileList()
        self.currFrame = self.fileList[self.currFrameIndex]
        self.currTime = 0

        # Instance variables to hold outcome data
        self.info = dict()

    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()

        # set up gui window and callback
        self.locGui = self._buildMainImage()
        cv2.imshow("Main", self.locGui)
        cv2.setMouseCallback("Main", self._mouseMainResponse)  # Set heading

        # set up map window and callback
        self.origMap = self._getOlinMap()
        (self.mapHgt, self.mapWid, dep) = self.origMap.shape
        self.currMap = self.origMap
        self._updateMap()  # to update the map position of the first frame
        cv2.imshow("Map", self.currMap)

        # run main loop until user quits or run out of frames
        ch = ' '
        while ch != 'q':
            x = cv2.waitKey(20)
            ch = chr(x & 0xFF)

        # close things down nicely
        self._writeData()
        cv2.destroyAllWindows()

    def _writeData(self):
        """Write the data collected to a timestamped file."""
        try:
            os.makedirs(self.outputFilePath)
        except:
            pass
        logName = time.strftime("Data-%b%d%Y-%H%M%S.txt")
        print(logName)
        fileOpen = False
        logFile = None
        try:
            logFile = open(self.outputFilePath + logName, 'w')
            fileOpen = True
        except:
            print("FAILED TO OPEN DATA FILE")

        for clickNum in self.info:
            [curTime, x, y, h] = self.info[clickNum]
            currCell = self.olinMap.convertLocToCell([x, y])
            dataStr = str(clickNum) + " " + str(curTime) + " " + str(x) + " " + str(y) + " " + str(h) + " " + str(
                currCell) + "\n"
            if fileOpen:
                logFile.write(dataStr)
            # print("Frame", clickNum, "with location", (x, y, h), "and time", curTime)
            # print("Cell ", currCell)
            # print(dataStr)
            print("Cell ", currCell)
        logFile.close()

    def _setupWindows(self):
        "Creates two windows and moves them to the right places on screen."
        cv2.namedWindow("Main")
        cv2.namedWindow("Map")
        cv2.moveWindow("Main", 0, 500)
        cv2.moveWindow("Map", 200, 0)

    def _buildMainImage(self):
        """Builds the gui window that displays the current frame values and has buttons for next and previous file.
        It also has a button to write the image data onto a txt file."""

        babyPink = (218, 195, 240)
        components = ("(" + self.currFrame[0] + ", " + self.currFrame[1] + ")  heading: " + self.currFrame[2] +
                      "  cell: " + self.currFrame[3])

        newMain = np.zeros((200, 480, 3), np.uint8)

        cv2.rectangle(newMain, (50, 100), (150, 150), babyPink, -1)
        cv2.putText(newMain, "Previous", (60, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (170, 100), (270, 150), babyPink, -1)
        cv2.putText(newMain, "Next", (180, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (290, 100), (430, 150), babyPink, -1)
        cv2.putText(newMain, "Write Image Data", (300, 130), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.putText(newMain, "%s" % components, (70, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
        return newMain

    def _mouseMainResponse(self, event, x, y, flags, param):
        """A mouse callback function that reads the user's click and responds by updating the robot's heading,
        or by moving on to the next frames in the video."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if (50 <= x < 150) and (100 <= y < 150):
                # click was in "previous" button
                if self.currFrameIndex != 0:
                    self.currFrameIndex -= 1
                self._updateGui()
                self._updateMap()
            elif (170 <= x < 270) and (100 <= y < 150):
                # click was in "next" button
                self.currFrameIndex += 1
                self._updateGui()
                self._updateMap()
            elif (290 <= x < 430) and (100 <= y < 150):
                # click was in "Write Image Data" button
                print("Not implemented yet")

    def _getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement,
        perhaps."""
        self.olinMap.cleanMapImage(cells=True, drawCellNum=True)  # draws the cells and their numbers on map
        # self.olinMap.drawPose((20, 100, 90), 15, (0, 255, 0)) # draws a point in the map
        origMap = self.olinMap.currentMapImg
        return origMap

    def _updateMap(self):
        """Updates the Olin Map to highlight the location of the current frame"""
        newLoc = self._convertWorldToMap(float(self.currFrame[0]), float(self.currFrame[1]))
        self.currMap = self._highlightLocation(newLoc, "pixels")
        cv2.imshow("Map", self.currMap)

    def _updateGui(self):
        """Updates the main GUI to match the values of the current frame"""
        self.currFrame = self.fileList[self.currFrameIndex]
        print(self.currFrame)
        self.locGui = self._buildMainImage()
        cv2.imshow("Main", self.locGui)

    def _highlightLocation(self, currPos, units="meters"):
        """Hightlight the robot's current location based on the input.
        currPos is given in meters, so need to convert: 20 pixels per meter."""
        newMap = self.origMap.copy()
        if units == "meters":
            (currX, currY) = currPos
            (mapX, mapY) = self._convertWorldToMap(currX, currY)
        else:
            (mapX, mapY) = currPos
        cv2.circle(newMap, (mapX, mapY), 10, (0, 255, 0), 10)
        return newMap

    def _convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return (mapXMeters, mapYMeters)

    def _convertWorldToMap(self, worldX, worldY):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        pixelX = worldX * 20.0
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.mapWid - 1 - pixelY
        mapY = self.mapHgt - 1 - pixelX
        return (int(mapX), int(mapY))

    def _getDataFileList(self):
        """ Returns a list of lists. Each of these lists represent a frame and every element is a component of
        every file name, that is, timestamp, x-coordinate, y-coordinate, heading, and cell."""
        folder_path = "../../res/locdata2022/"
        input_path = os.listdir(folder_path)
        fileList = []
        for text_file in input_path:
            with open(folder_path + text_file) as textFile:
                for line in textFile:
                    lineWords = []
                    words = line.split(" ")
                    words[5] = words[5].replace("\n", "")
                    for i in range(2, 6):
                        lineWords.append(words[i])
                    fileList.append(lineWords)
        return fileList


if __name__ == "__main__":
    reviewer = ImageReview(output_file_path=basePath + "res/locdata2022/")
    reviewer.go()
