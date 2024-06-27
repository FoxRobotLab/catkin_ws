# coding=utf-8
"""--------------------------------------------------------------------------------------------------------------------
This program was built off the bones of the locsForFrames program. This program allows a user to follow the robot and mark
the robot's location in real time. The click number, time when the "next" button is pressed, x-coordinate, y-coordinate,
and heading are all read into a text file. This program features two GUIs: one is a map the user can click on in order
to record location, and the other is a GUI that shows the x-coordinate, y-coordinate, heading, and the number click the
user is on.

Author: Malini Sharma
Updated July 2022 by Bea Bautista, Yifan Wu
--------------------------------------------------------------------------------------------------------------------"""

import os
import time
import datetime

import cv2
import numpy as np

from src.match_seeker.scripts.olri_classifier.OlinWorldMap import WorldMap
# from src.match_seeker.scripts.olri_classifier.paths import DATA
from src.match_seeker.scripts.olri_classifier.DataPaths import basePath


class RealTimeLocs(object):

    def __init__(self, outputFilePath):
        """Set up data to be held, including displayed and stored maps, current labeling location
        and heading, and dictionary mapping click numbers to times, locations, and headings."""
        self.currLoc = (0, 0)
        self.currHeading = 0

        # variables for managing data source
        self.clickNum = 1
        self.outputFilePath = outputFilePath

        # Olin Map
        self.olinMap = WorldMap()

        # instance variables to hold displayed images
        self.locGui = None
        self.origMap = None
        self.currMap = None
        self.currFrame = None
        self.currTime = 0

        # Instance variables to hold outcome data
        self.info = dict()


    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()

        # set up gui window and callback
        self.locGui = self._buildMainImage()
        self._displayStatus()
        cv2.imshow("Main", self.locGui)
        cv2.setMouseCallback("Main", self._mouseMainResponse)   # Set heading

        # set up map window and callback
        self.origMap = self._getOlinMap()
        (self.mapHgt, self.mapWid, dep) = self.origMap.shape
        self.currMap = self.origMap
        cv2.imshow("Map", self.currMap)
        cv2.setMouseCallback("Map", self._mouseSetLoc)          # Set x, y location

        # run main loop until user quits or run out of frames
        ch = ' '
        while ch != 'q':
            x = cv2.waitKey(20)
            ch = chr(x & 0xFF)
            if self.currLoc != (0, 0): # if robot location has been placed on the map
                (mapX, mapY) = self._convertWorldToMap(self.currLoc[0], self.currLoc[1])
                if ch in "wasd":
                    if ch == 'a':
                        mapX -= 2
                    elif ch == 'd':
                        mapX += 2
                    elif ch == 'w':
                        mapY -= 2
                    elif ch == 's':
                        mapY += 2
                    self.currLoc = self._convertMapToWorld(mapX, mapY)
                    self._updateMap( (mapX, mapY) )

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
            print ("FAILED TO OPEN DATA FILE")

        for clickNum in self.info:
            [curTime, x, y, h] = self.info[clickNum]
            currCell = self.olinMap.convertLocToCell([x, y])
            dataStr = str(clickNum) + " " + str(curTime) + " " +str(x) + " " + str(y) + " " + str(h) + " " + str(currCell) + "\n"
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
        cv2.moveWindow("Main", 30, 50)
        cv2.moveWindow("Map", 700, 50)


    def _buildMainImage(self):
        """Builds the gui window image with buttons at given locations. Note that space is left for
        displaying the robot's current location, but that is done by a separate method."""

        dustyRose = (186, 149, 219)
        deepPink = (150, 105, 192)
        babyPink = (218, 195, 240)

        newMain = np.zeros((420, 210, 3), np.uint8)

        # Put squares for NE, SE, SW, NW
        info = [[0, 0, "NW 45"], [0, 140, "SW 135"], [140, 0, "NE 315"], [140, 140, "SE 225"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), dustyRose, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        # Put squares for N, S, E, W
        info = [[70, 0, "N 0"], [70, 140, "S 180"], [0, 70, "W 90"], [140, 70, "E 270"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), deepPink, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (20, 350), (99, 400), babyPink, -1)
        cv2.putText(newMain, "Previous", (30, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (110, 350), (189, 400), babyPink, -1)
        cv2.putText(newMain, "Next", (135, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
        return newMain

        # cv2.rectangle(newMain, (62, 350), (141, 400), babyPink, -1)
        # cv2.putText(newMain, "Next", (85, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
        # return newMain


    def _displayStatus(self):
        """Displays the current location and saved loc counter on the main window."""

        white = (255, 255, 255)
        cv2.rectangle(self.locGui, (0, 210), (210, 340), (0, 0, 0), -1)
        floatTemplate = "{0:s} = {1:^6.2f}"
        intTemplate = "{0:s} = {1:^3d}"
        xStr = floatTemplate.format('x', self.currLoc[0])
        yStr = floatTemplate.format('y', self.currLoc[1])
        hStr = intTemplate.format('yaw', self.currHeading)

        countStr = intTemplate.format('saved loc', self.clickNum)
        cv2.putText(self.locGui, "Current location:", (20, 240), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGui, xStr, (30, 260), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGui, yStr, (30, 280), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGui, hStr, (30, 300), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGui, countStr, (20, 330), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.imshow("Main", self.locGui)


    def _mouseMainResponse(self, event, x, y, flags, param):
        """A mouse callback function that reads the user's click and responds by updating the robot's heading,
        or by moving on to the next frames in the video."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if (0 <= x < 70) and (0 <= y < 70):
                # click was in "northwest" heading square
                self.currHeading = 45
            elif (0 <= x < 70) and (70 <= y < 140):
                # click was in "west" heading square
                self.currHeading = 90
            elif (0 <= x < 70) and (140 <= y < 210):
                # click was in "southwest" heading square
                self.currHeading = 135
            elif (70 <= x < 140) and (0 <= y < 70):
                # click was in "north" heading square
                self.currHeading = 0
            elif (70 <= x < 140) and (140 <= y < 210):
                # click was in "south" heading square
                self.currHeading = 180
            elif (140 <= x < 210) and (0 <= y < 70):
                # click was in "northeast" heading square
                self.currHeading = 315
            elif (140 <= x < 210) and (70 <= y < 140):
                # click was in "east" heading square
                self.currHeading = 270
            elif (140 <= x < 210) and (140 <= y < 210):
                # click was in "southeast" heading square
                self.currHeading = 225
            elif (20 <= x < 100) and (350 <= y <= 400):
                # click was in "Previous" frames
                self.clickNum = self.clickNum - 1
                self.getValues()
            #elif (62 <= x < 141) and (350 <= y <= 400):
            elif (110 <= x < 190) and (350 <= y <= 400):
                # click was in "Next" frames
                time1 = time.localtime()
                self.getValues()
                self.clickNum = self.clickNum + 1

            self._displayStatus()

    def getValues(self):
        # time1 = time.localtime()
        # self.currTime = time.strftime("%H:%M:%S", time1)
        time1 = datetime.datetime.now()
        # self.currTime = datetime.datetime.strftime(time1, "%H:%M:%S:%f")
        self.currTime = "{}".format(time.strftime("%Y%m%d-%H:%M:%S"))
        self.info[self.clickNum] = [self.currTime, self.currLoc[0], self.currLoc[1], self.currHeading]

        currCell = self.olinMap.convertLocToCell([self.currLoc[0], self.currLoc[1]])
        print("Cell ", currCell)

    def _getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        self.olinMap.cleanMapImage(cells=True, drawCellNum=True)
        origMap = self.olinMap.currentMapImg
        return origMap


    def _mouseSetLoc(self, event, x, y, flags, param):
        """A mouse callback function that reads the user's click location and marks it on the map, also making it
        available to the program to use."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.currLoc = self._convertMapToWorld(x, y)
            print("mouseSetLoc self.currLoc =", self.currLoc)
            self._updateMap( (x, y) )


    def _updateMap(self, mapLoc):
        self.currMap = self._highlightLocation(mapLoc, "pixels")
        cv2.imshow("Map", self.currMap)
        self._displayStatus()


    def _highlightLocation(self, currPos, units = "meters"):
        """Hightlight the robot's current location based on the input.
        currPos is given in meters, so need to convert: 20 pixels per meter."""
        newMap = self.origMap.copy()
        if units == "meters":
            (currX, currY) = currPos
            (mapX, mapY) = self._convertWorldToMap(currX, currY)
        else:
            (mapX, mapY) = currPos
        cv2.circle(newMap, (mapX, mapY), 7, (0, 140, 0), -1)
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


if __name__ == "__main__":
    # realTimer = RealTimeLocs(outputFilePath= DATA + "testWalkandTimestamp/")
    realTimer = RealTimeLocs(outputFilePath= basePath + "res/locdata2022/")
    realTimer.go()

