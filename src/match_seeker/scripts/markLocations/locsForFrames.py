# coding=utf-8
"""--------------------------------------------------------------------------------------------------------------------
The first step of location tagging. It can be run on video (if you want to tag locations before removing duplicate
images) or on a folder of jpegs. This program displays a map, an image, and a grid of possible angular headings.
For each image you can adjust the robot’s location on the map with the WASD keys, adjust its heading by clicking on
the grid, and advance with the "Next" button. The robot cameras have fish eye lenses, so the always appear to be a few
feet ahead of where they actually are. Typically, if the base of a wall is along the bottom of the image, the robot is
in the center of the hallway. If a doorframe you are driving past is close to the edge of the image, the robot is in the
middle of that doorway.

Modified Jun 1 2018: Added "Previous" button and modified "Next Frame" button (size, name to "Next").
--------------------------------------------------------------------------------------------------------------------"""

import os
import time

import cv2
import numpy as np

# import readMap
import src.match_seeker.scripts.markLocations.readMap as readMap


class LabeledFrames(object):

    def __init__(self, mapFile, dataSource, outputFilePath, mode = "image"):
        """Set up data to be held, including displayed and stored maps, current labeling location
        and heading, and dictionary mapping frames numbers to locations and headings. Default mode
        is to run on a stored video feed, but it can also run through a folder of images. The "data" field
        is either the name of the video or of the folder where the images are found. The mode variable is either
        'video' or 'images'"""
        self.currLoc = (0, 0)
        self.currHeading = 0
        self.mode = mode

        # variables for managing data source
        #self.videoObject = None     # only used if mode = video
        self.imgFileList = []    # only used if mode = 'images'
        self.picNum = -1
        self.mapFilename = mapFile
        self.dataSource = dataSource
        self.dataDone = False
        self.imgIndex = 0
        self.outputFilePath = outputFilePath

        # instance variables to hold displayed images
        self.locGUI = None
        self.origMap = None
        self.currMap = None
        self.currImg = None

        # Instance variables to hold outcome data
        self.labeling = dict()


    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()

        # set up main window and callback
        self.locGUI = self._buildLocGUI()
        self._displayStatus()
        cv2.imshow("GUI", self.locGUI)
        cv2.setMouseCallback("GUI", self._mouseMainResponse)   # Set heading

        # set up map window and callback
        self.origMap = self._getOlinMap()
        (self.mapHgt, self.mapWid, dep) = self.origMap.shape
        self.currMap = self.origMap
        cv2.imshow("Map", self.currMap)
        cv2.setMouseCallback("Map", self._mouseSetLoc)          # Set x, y location

        # set up video capture and first frames
        self._setupImageCapture()
        ch = ' '
        nextFrame = self._getNextImage()
        if not nextFrame:
            self.dataDone = True

        # run main loop until user quits or run out of frames
        while (not self.dataDone) and (ch != 'q'):
            cv2.putText(self.currImg, str(self.picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Image", self.currImg)
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
                nextFrame = self._getNextImage()
                if not nextFrame:
                    self.dataDone = True

        # close things down nicely
        self._writeData()
        self._cleanUp()
        cv2.destroyAllWindows()


    def _writeData(self):
        """Write the data collected to a timestamped file."""
        try:
            os.makedirs(self.outputFilePath)
        except:
            pass
        logName = time.strftime("Data-%b%d%a-%H%M%S.txt")
        print(logName)
        fileOpen = False
        logFile = None
        try:
            logFile = open(self.outputFilePath + logName, 'w')
            fileOpen = True
        except:
            print ("FAILED TO OPEN DATA FILE")

        for picNum in self.labeling:
            [x, y, yaw] = self.labeling[picNum]
            dataStr = str(picNum) + " " + str(x) + " " + str(y) + " " + str(yaw) + "\n"
            if fileOpen:
                logFile.write(dataStr)
            print("Frame", picNum, "with location", (x, y, yaw))
        logFile.close()


    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("GUI")
        cv2.namedWindow("Image")
        cv2.namedWindow("Map")
        cv2.moveWindow("GUI", 30, 50)
        cv2.moveWindow("Image", 30, 550)
        cv2.moveWindow("Map", 700, 50)


    def _buildLocGUI(self):
        """Builds the main window image with buttons at given locations. Note that space is left for
        displaying the robot's current location, but that is done by a separate method."""

        lightBlue1 = (255, 226, 176)
        quartz = (243, 217, 217)
        lightBlue2 = (238, 211, 164)

        newMain = np.zeros((420, 210, 3), np.uint8)

        # Put squares for NE, SE, SW, NW
        info = [[0, 0, "NW 45"], [0, 140, "SW 135"], [140, 0, "NE 315"], [140, 140, "SE 225"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), lightBlue1, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        # Put squares for N, S, E, W
        info = [[70, 0, "N 0"], [70, 140, "S 180"], [0, 70, "W 90"], [140, 70, "E 270"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), quartz, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (20, 350), (99, 400), lightBlue2, -1)
        cv2.putText(newMain, "Previous", (30, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (110, 350), (189, 400), lightBlue2, -1)
        cv2.putText(newMain, "Next", (135, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
        return newMain


    def _displayStatus(self):
        """Displays the current location and frames counter on the main window."""

        white = (255, 255, 255)
        cv2.rectangle(self.locGUI, (0, 210), (210, 340), (0, 0, 0), -1)
        floatTemplate = "{0:s} = {1:^6.2f}"
        intTemplate = "{0:s} = {1:^3d}"
        xStr = floatTemplate.format('x', self.currLoc[0])
        yStr = floatTemplate.format('y', self.currLoc[1])
        yawStr = intTemplate.format('yaw', self.currHeading)

        countStr = intTemplate.format('frames', self.picNum)
        cv2.putText(self.locGUI, "Current location:", (20, 240), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGUI, xStr, (30, 260), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGUI, yStr, (30, 280), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGUI, yawStr, (30, 300), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.putText(self.locGUI, countStr, (20, 330), cv2.FONT_HERSHEY_PLAIN, 1.0, white)
        cv2.imshow("Main", self.locGUI)


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
                self.imgIndex -= 1
                self._processToNextFrame()
            elif (110 <= x < 190) and (350 <= y <= 400):
                # click was in "Next" frames
                if self.imgIndex >= len(self.imgFileList):
                    self.dataDone = True
                else:
                    self.imgIndex += 1
                    self._processToNextFrame()

            self._displayStatus()

    def _processToNextFrame(self):
        self.labeling[self.picNum] = [self.currLoc[0], self.currLoc[1], self.currHeading]
        mapX, mapY = self._convertWorldToMap(self.currLoc[0], self.currLoc[1])
        self._updateMap((mapX, mapY))
        nextFrame = self._getNextImage()
        if not nextFrame:
            print("WHOOPS!, no image found!")
            self.dataDone = True
        else:
            cv2.putText(self.currImg, str(self.picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Image", self.currImg)
        if self.picNum in self.labeling.keys():
            x, y, yaw = self.labeling[self.picNum]
            self.currLoc = (x, y)
            self.currHeading = yaw
            mapX, mapY = self._convertWorldToMap(self.currLoc[0], self.currLoc[1])
            self._updateMap((mapX, mapY))


    def _getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(self.mapFilename, 20)
        map2 = np.flipud(origMap)
        olinMap = np.rot90(map2)
        return olinMap


    def _mouseSetLoc(self, event, x, y, flags, param):
        """A mouse callback function that reads the user's click location and marks it on the map, also making it
        available to the program to use."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.currLoc = self._convertMapToWorld(x, y)
            print "mouseSetLoc self.currLoc =", self.currLoc
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
        cv2.circle(newMap, (mapX, mapY), 6, (0, 0, 255))
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



    def _setupImageCapture(self):
        """Sets up the video capture for the given filename, printing a message if it failed
        and returning None in that case."""
        # if self.mode == 'video':
        #     try:
        #         self.videoObject = cv2.VideoCapture(self.dataSource) ##I don't think this is being used correctly -> check with Susan
        #     except:
        #         print("Capture failed!")
        #         self.videoObject = None
        # else:
        self.imgFileList = os.listdir(self.dataSource)
        self.imgFileList.sort()


    def _getNextImage(self):
        """Gets the next frames from the camera or from reading the next file"""
        # if self.mode == 'video':
        #     good, frames = self.videoObject.read()
        #     if not good:
        #         return good
        #     self.picNum += 1
        #     self.currImg = frames
        #     return good
        # else:
        while True:
            if self.imgFileList == []:
                return False
            if self.imgIndex >= len(self.imgFileList):
                self.dataDone = True
                break
            filename = self.imgFileList[self.imgIndex]
            if filename[-3:] in {"jpg", "png"}:
                try:
                    newIm = cv2.imread(self.dataSource + filename)
                except IOError:
                    return False
                thisNum = self._extractNum(filename)
                self.currImg = newIm
                self.picNum = thisNum
                return True
            else:
                # Check for non-jpg or -png file (e.x.: .DS_Store)
                print("Non-jpg or -png file in folder: ", filename, "... skipping!")
                self.imgFileList.remove(filename)
                continue


    def _extractNum(self, fileString):
        """finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for num in fileString:
            if num in '0123456789':
                foundDigits = True
                numStr += num
            elif foundDigits == True:
                break
        if numStr != "":
            return int(numStr)
        else:
            self.picNum += 1
            return self.picNum


    # def _cleanUp(self):
    #     """Closes down video capture, if necessary"""
    #     if self.mode == 'video':
    #         self.videoObject.release()



if __name__ == "__main__":
    # frameRecorder = LabeledFrames("olinNewMap.txt", "../../res/Videos/may30.avi", 'video')
    # frameRecorder.go()
    catkinPath = "/home/macalester/"
    # catkinPath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/"
    basePath = "PycharmProjects/catkin_ws/src/match_seeker/"

    frameRecorder = LabeledFrames(mapFile=catkinPath + basePath + "res/map/olinNewMap.txt",
                                  dataSource="/home/macalester/allFrames/",
                                  outputFilePath= "/home/macalester/testframes/",
                                  mode="images")

    frameRecorder.go()
