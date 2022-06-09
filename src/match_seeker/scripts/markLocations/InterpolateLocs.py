# coding=utf-8
"""--------------------------------------------------------------------------------------------------------------------
The first step of location tagging. It can be run on video (if you want to tag locations before removing duplicate
images) or on a folder of jpegs. This program displays a map, an image, and a grid of possible angular headings.
For each image you can adjust the robot’s location on the map with the WASD keys, adjust its heading by clicking on
the grid, and advance with the "Next" button. The robot cameras have fish eye lenses, so the always appear to be a few
feet ahead of where they actually are. Typically, if the base of a wall is along the bottom of the image, the robot is
in the center of the hallway. If a doorframe you are driving past is close to the edge of the image, the robot is in the
middle of that doorway.

This also takes into account the text file of pre-tagged images. It combines tagged and untagged images. It does this by
displaying tagged images on the map, highlighting both the previous and next tagged locations.

Modified Jun 1 2018: Added "Previous" button and modified "Next Frame" button (size, name to "Next").

Author: Jane Pellegrini
--------------------------------------------------------------------------------------------------------------------"""

import os
import time

import cv2
import numpy as np
import math


# import readMap
import src.match_seeker.scripts.markLocations.readMap as readMap

#TODO: add an autoInterpollate button
class Interpolator(object):

    def __init__(self, mapFile, dataSource, outputFilePath,inputLocsFilePath, mode = "image"):
        """Set up data to be held, including displayed and stored maps, current labeling location
        and heading, and dictionary mapping frames numbers to locations and headings. Default mode
        is to run on a stored video feed, but it can also run through a folder of images. The "data" field
        is either the name of the video or of the folder where the images are found. The mode variable is either
        'video' or 'images'"""
        self.currLoc = (0, 0)
        self.currHeading = 0
        self.mode = mode

        # variables for managing data source
        self.imgFileList = []    # only used if mode = 'images'

        self.picNum = -1
        self.mapFilename = mapFile
        self.dataSource = dataSource
        self.dataDone = False
        self.imgIndex = 0
        self.outputFilePath = outputFilePath
        self.inputLocsFilePath = inputLocsFilePath

        # instance variables to hold displayed images
        self.mainImg = None
        self.origMap = None
        self.currMap = None
        self.currFrame = None


        # Instance variables to hold outcome data
        self.labeling = dict()


        self.previousStamp = None
        self.nextStamp = None


    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()

        # set up main window and callback
        self.mainImg = self._buildMainImage()
        self._displayStatus()
        cv2.imshow("Main", self.mainImg)
        cv2.setMouseCallback("Main", self._mouseMainResponse)   # Set heading


        # set up map window and callback
        self.origMap = self._getOlinMap()
        (self.mapHgt, self.mapWid, dep) = self.origMap.shape
        self.currMap = self.origMap
        cv2.imshow("Map", self.currMap)
        cv2.setMouseCallback("Map", self._mouseSetLoc)          # Set x, y location

        # set up data from walking files
        file = open(self.inputLocsFilePath, "r")
        walking = file.readlines()
        self.walkingNums = []
        self.walking = {}
        file.close()
        for line in walking:
            self.walkingNums.append(int(line.split()[0]))  # This assumes the text file will be in a certain format
            lineList= line.split()
            #(x,y) = self._convertMapToWorld(float(lineList[1]),float(lineList[2]))
            self.walking[lineList[0]]= [float(lineList[1]), float(lineList[2]), int(lineList[3])]
        self.origMap = self.drawWalkingLocations(self.walking)     # draws the circles for the already marked locations


        # set up video capture and first frames
        self._setupImageCapture()
        ch = ' '
        goodFrame = self._getNextImage()
        if not goodFrame:
            self.dataDone = True

        self.fileNumberList = []
        for file in self.imgFileList:
            filenumber= self._extractNum(file)
            self.fileNumberList.append(filenumber)

        # run main loop until user quits or run out of frames
        while (not self.dataDone) and (ch != 'q'):
            cv2.putText(self.currFrame, str(self.picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Image", self.currFrame)
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
                goodFrame = self._getNextImage()
                if not goodFrame:
                    self.dataDone = True

        # close things down nicely
        self._writeData()
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
        for filename in self.imgFileList:
            if filename[-3:] in {"jpg", "png"}:
                try:
                    newIm = cv2.imread(self.dataSource + filename)
                except IOError:
                    print("error reading jpg file")
                thisNum = self._extractNum(filename)
                written = False
                if thisNum in self.walkingNums:
                    [x, y, h] = self.walking[str(thisNum)]
                    dataStr = str(thisNum) + " " + str(x) + " " + str(y) + " " + str(h) + "\n"
                    written = True
                elif thisNum in self.labeling:
                    [x, y, h] = self.labeling[thisNum]
                    dataStr = str(thisNum) + " " + str(x) + " " + str(y) + " " + str(h) + "\n"
                    written = True
                if fileOpen and written == True:
                    logFile.write(dataStr)
                    print("Frame", thisNum, "with location", (x, y, h))
            else:
                print("not printing non-jpg")

        logFile.close()


    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("Main")
        cv2.namedWindow("Image")
        cv2.namedWindow("Map")
        cv2.moveWindow("Main", 30, 50)
        cv2.moveWindow("Image", 30, 550)
        cv2.moveWindow("Map", 700, 50)


    def _buildMainImage(self):
        """Builds the main window image with buttons at given locations. Note that space is left for
        displaying the robot's current location, but that is done by a separate method."""

        yellow = (0, 255, 255)
        cyan = (255, 255, 0)
        green = (0, 255, 0)

        newMain = np.zeros((480, 210, 3), np.uint8)

        # Put squares for NE, SE, SW, NW
        info = [[0, 0, "NW 45"], [0, 140, "SW 135"], [140, 0, "NE 315"], [140, 140, "SE 225"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), yellow, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        # Put squares for N, S, E, W
        info = [[70, 0, "N 0"], [70, 140, "S 180"], [0, 70, "W 90"], [140, 70, "E 270"]]
        for square in info:
            (x, y, strng) = square
            cv2.rectangle(newMain, (x, y), (x + 69, y + 69), cyan, -1)
            cv2.putText(newMain, strng, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (20, 350), (99, 400), green, -1)
        cv2.putText(newMain, "Previous", (30, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (110, 350), (189, 400), green, -1)
        cv2.putText(newMain, "Next", (135, 380), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))

        cv2.rectangle(newMain, (20, 410),(99, 450), green, -1)
        cv2.putText(newMain, "Auto Line", (30, 430), cv2.FONT_HERSHEY_PLAIN, .8, (0,0,0))      #BOUNDARIES OF AUTO LINE

        cv2.rectangle(newMain, (110, 410), (189, 450), green , -1)
        cv2.putText(newMain, "Auto Spin", (120, 430), cv2.FONT_HERSHEY_PLAIN, .8, (0,0,0))      #BOUNDARIES OF AUTO SPIN

        return newMain


    def _displayStatus(self):
        """Displays the current location and frames counter on the main window."""

        yellow = (0, 255, 255)
        cv2.rectangle(self.mainImg, (0, 210), (210, 340), (0, 0, 0), -1)
        floatTemplate = "{0:s} = {1:^6.2f}"
        intTemplate = "{0:s} = {1:^3d}"
        xStr = floatTemplate.format('x', self.currLoc[0])
        yStr = floatTemplate.format('y', self.currLoc[1])
        hStr = intTemplate.format('h', self.currHeading)

        countStr = intTemplate.format('frames', self.picNum)
        cv2.putText(self.mainImg, "Current location:", (20, 240), cv2.FONT_HERSHEY_PLAIN, 1.0, yellow)
        cv2.putText(self.mainImg, xStr, (30, 260), cv2.FONT_HERSHEY_PLAIN, 1.0, yellow)
        cv2.putText(self.mainImg, yStr, (30, 280), cv2.FONT_HERSHEY_PLAIN, 1.0, yellow)
        cv2.putText(self.mainImg, hStr, (30, 300), cv2.FONT_HERSHEY_PLAIN, 1.0, yellow)
        cv2.putText(self.mainImg, countStr, (20, 330), cv2.FONT_HERSHEY_PLAIN, 1.0, yellow)
        cv2.imshow("Main", self.mainImg)


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
            elif (20 <= x <= 99) and (410 <=y <= 450):
                self.automaticInterpolateLine()
            elif (110 <= x <= 189) and (410 <= y <= 450):
                self.automaticInterpolateSpin()

            self._displayStatus()

    def automaticInterpolateSpin(self):
        """
        Given two points with relatively the same location that have the same yaw, split all the images up and assign
        them automatically interpolated location and yaw.
        :return:
        """
        if self.previousStamp != None and self.nextStamp != None:
            numprev = str(self.walkingNums[self.previousStamp])
            infoprev = self.walking[numprev]
            print(infoprev)
            numnext = str(self.walkingNums[self.nextStamp])
            infonext = self.walking[numnext]
            autoFrameNums = []
            for framenum in self.fileNumberList:
                if framenum < self.walkingNums[self.nextStamp] and framenum > self.walkingNums[self.previousStamp]:
                    autoFrameNums.append(framenum)
            numAutoFrames = len(autoFrameNums)
            xPrev = float(infoprev[0])
            yPrev = float(infoprev[1])
            xNext = float(infonext[0])
            yNext = float(infonext[1])
            xAvg = (xPrev + xNext) / 2
            yAvg = (yPrev + yNext) / 2
            yawPrev = float(infoprev[2])
            yawNext = float(infonext[2])
            yawChange = 360.0 / numAutoFrames      # possibly could improve by giving the option of finding the difference
            print(type(yawChange))
            currYaw = yawPrev
            for frameNum in autoFrameNums:
                currYaw = currYaw- yawChange
                if currYaw < 0:
                    currYaw = currYaw + 360
                thisYaw =self.roundToNearestAngle(currYaw)
                self.labeling[frameNum] = [xAvg, yAvg, thisYaw]
            # move to the next unlabeled frames
            self.currLoc = (xNext, yNext)
            self.currHeading = int(yawNext)
            # here is where picNum needs to be updated
            self.imgIndex += numAutoFrames
            i = self.fileNumberList.index(self.walkingNums[self.nextStamp])
            self.picNum = self.fileNumberList[i + 1]
            filename = self.imgFileList[i + 1]
            mapX, mapY = self._convertWorldToMap(xNext, yNext)
            self._updateMap((mapX, mapY))
            try:
                newIm = cv2.imread(self.dataSource + filename)
            except IOError:
                return False
            self.currFrame = newIm
            cv2.imshow("Image", self.currFrame)
        else:
            print("Cannot Interpolate: Not between two stamps")

    def roundToNearestAngle(self, angle):
        """
        Takes in an angle and returns the angle within the compass (separated by

        :param angle:
        :return:
        """
        listOfAngles = [0, 45, 90, 135, 180, 225, 270, 315]
        dif = 1000000000
        smallestAngle = 0
        for currAngle in listOfAngles:
            currDif = abs(currAngle - angle)
            if currDif < dif:
                dif = currDif
                smallestAngle = currAngle
        return smallestAngle

    def automaticInterpolateLine(self):
        """
        Finds the two adjacent marked locations, asks for a desired yaw, and then evenly interpolates the images
        :return:
        """
        if self.previousStamp != None and self.nextStamp != None:
            num = str(self.walkingNums[self.previousStamp])
            info = self.walking[num]
            xPrev = float(info[0])                          # maybe change into ints instead?
            yPrev = float(info[1])
            num = str(self.walkingNums[self.nextStamp])
            info = self.walking[num]
            xNext = float(info[0])
            yNext = float(info[1])
            autoFrameNums = []
            for framenum in self.fileNumberList:
                if framenum < self.walkingNums[self.nextStamp] and framenum > self.walkingNums[self.previousStamp]:
                    autoFrameNums.append(framenum)
            numAutoFrames = len(autoFrameNums)
            xChange = (xNext - xPrev) / numAutoFrames
            yChange = (yNext - yPrev) / numAutoFrames
            x = xPrev
            y = yPrev
            yaw = self.calculateYaw(xChange, yChange)
            for frameNum in autoFrameNums:
                x += xChange
                y += yChange
                thisX = float("{0:.1f}".format(x))
                thisY = float("{0:.1f}".format(y))
                self.labeling[frameNum] = [thisX, thisY, yaw]
            # move to the next unlabeled frames
            thisX = float("{0:.1f}".format(x))
            thisY = float("{0:.1f}".format(y))
            self.currLoc = (thisX, thisY)
            self.currHeading = int(yaw)
            #here is where picNum needs to be updated
            self.imgIndex += numAutoFrames
            i=self.fileNumberList.index(self.walkingNums[self.nextStamp])
            self.picNum = self.fileNumberList[i+1]
            filename = self.imgFileList[i+1]
            mapX, mapY =self._convertWorldToMap(thisX, thisY)
            self._updateMap((mapX, mapY))
            try:
                newIm = cv2.imread(self.dataSource + filename)
            except IOError:
                return False
            self.currFrame = newIm
            cv2.imshow("Image", self.currFrame)
        else:
            print("Cannot Interpolate: Not between two stamps")

    def calculateYaw(self, xChange, yChange):
        """
        Takes in a vector with an x and a y component, and calculates the angle counterclockwise with the positive
        y axis
        :param xChange:
        :param yChange:
        :return:
        """
        angle = math.atan2(yChange,xChange)        #flipped x and y because the map has flipped x and y
        angle = math.degrees(angle)
        newAngle = 0
        if angle < 0:              #checks if the angle is negative, converts to the positive version
            angle += 360
        yaw =0
        # now calculate which category this angle is in
        if angle > 337.5 or angle < 22.5 :
            yaw = 0
        if angle >= 22.5 and angle <= 67.5:
            yaw  = 45
        if angle > 67.5 and angle < 112.5:
            yaw = 90
        if angle >= 112.5 and angle <= 157.5:
            yaw = 135
        if angle > 157.5 and angle < 202.5:
            yaw = 180
        if angle >= 202.5 and angle <= 247.5:
            yaw = 225
        if angle > 247.5 and angle < 292.5:
            yaw = 270
        if angle >= 292.5 and angle <= 337.5:
            yaw = 315
        return yaw



    def _processToNextFrame(self):
        self.labeling[self.picNum] = [self.currLoc[0], self.currLoc[1], self.currHeading]
        mapX, mapY = self._convertWorldToMap(self.currLoc[0], self.currLoc[1])
        self._updateMap((mapX, mapY))
        goodFrame = self._getNextImage()
        if not goodFrame:
            print("WHOOPS!, no image found!")
            self.dataDone = True
        else:
            cv2.putText(self.currFrame, str(self.picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Image", self.currFrame)
        if self.picNum in self.labeling.keys():
            x, y, h = self.labeling[self.picNum]
            self.currLoc = (x, y)
            self.currHeading = h
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
        currPos is given in meters, so need to convert: 20 pixels per meter.
        Also highlights the previous and next stamped frames"""
        newMap = self.origMap.copy()
        self.setCurrentStamps()
        # highlights the previous location in black
        if self.previousStamp != None:
            num = str(self.walkingNums[self.previousStamp])
            info = self.walking[num]
            (pcurrX, pcurrY) = (float(info[0]), float(info[1]))
            (mapX, mapY) = self._convertWorldToMap(pcurrX, pcurrY)
            cv2.circle(newMap, (mapX, mapY), 8, (255, 0, 0))
        # highlights the next location in green
        if self.nextStamp != None:
            num = str(self.walkingNums[self.nextStamp])
            info = self.walking[num]
            (ncurrX, ncurrY) = (float(info[0]), float(info[1]))
            (mapX, mapY) = self._convertWorldToMap(ncurrX, ncurrY)
            cv2.circle(newMap, (mapX, mapY), 8, (0, 255, 0))
        # highlighting the location of the pointer in red
        if units == "meters":
            (currX, currY) = currPos
            (mapX, mapY) = self._convertWorldToMap(currX, currY)
        else:
            (mapX, mapY) = currPos
        cv2.circle(newMap, (mapX, mapY), 6, (0, 0, 255))
        return newMap


    def setCurrentStamps(self):
        """
        determines the previous stamped location and the next stamped location to the current frames.
        :return:
        """
        if self.picNum < self.walkingNums[0]:
            self.previousStamp = None
            self.nextStamp= 1
        if self.picNum > self.walkingNums[len(self.walkingNums)-1]:
            self.previousStamp = len(self.walkingNums)-1
            self.nextStamp = None
        else:
            for i in range(len(self.walkingNums)-1):
                if self.picNum > self.walkingNums[i] and self.picNum < self.walkingNums[i+1]:
                    self.previousStamp = i
                    self.nextStamp = i+1


    def drawWalkingLocations(self, walkingDict):
        """ Draws a small black circle for all of the locations that have been marked using time- stamping"""
        newMap = self.origMap.copy()
        for loc in walkingDict:
            elems = walkingDict[loc]
            x= elems[0]
            y= elems[1]
            x,y = self._convertWorldToMap(x,y)
            cv2.circle(newMap, (x, y), 4, (0, 0, 0))
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
        self.imgFileList = os.listdir(self.dataSource)
        self.imgFileList.sort()


    def _getNextImage(self):
        """Gets the next frames from the camera or from reading the next file"""
        while True:
            if self.imgFileList == []:
                return False
            if self.imgIndex >= len(self.imgFileList):
                return False
            filename = self.imgFileList[self.imgIndex]
            if filename[-3:] in {"jpg", "png"}:
                try:
                    newIm = cv2.imread(self.dataSource + filename)
                except IOError:
                    return False
                thisNum = self._extractNum(filename)
                if thisNum in self.walkingNums:
                    #print("image already determined")
                    self.imgIndex +=1
                    self.currFrame = newIm
                    self.picNum = thisNum
                else:
                    self.currFrame = newIm
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
        for c in fileString:
            if c in '0123456789':
                foundDigits = True
                numStr += c
            elif foundDigits == True:
                break
        if numStr != "":
            return int(numStr)
        else:
            self.picNum += 1
            return self.picNum


if __name__ == "__main__":
    # frameRecorder = LabeledFrames("olinNewMap.txt", "../../res/Videos/may30.avi", 'video')
    # frameRecorder.go()
    #catkinPath = "/home/macalester/"
    # catkinPath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/"qqq
    #basePath = "PycharmProjects/catkin_ws/src/match_seeker/"

    # InterpolatorObj = Interpolator(mapFile=catkinPath + basePath + "res/map/olinNewMap.txt",
    #                               dataSource="/home/macalester/allFrames/",
    #                               outputFilePath= "/home/macalester/testframes/",
    #                               inputLocsFilePath="",
    #                               mode="images",
    #                               )


    # catkinPath = "/Users/johnpellegrini/"
    # basePath = "PycharmProjects/catkin_ws/src/match_seeker/"
    InterpolatorObj = Interpolator(mapFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/map/olinNewMap.txt",
                                  # dataSource= "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/july6Frames3/",
                                  dataSource="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/",
                                  outputFilePath= "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/",
                                  inputLocsFilePath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/july6MatchedCheckpoints3.txt",
                                  mode="images",
                                  )

    InterpolatorObj.go()
