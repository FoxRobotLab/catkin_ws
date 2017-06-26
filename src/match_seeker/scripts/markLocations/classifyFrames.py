"""
File: classifyFrames.py
Author: Susan Fox
Date: February 2017

This file allows the user to select among a set of options for describing the environment the picture is in.
Categories:
 0 In Room  (just a view of a random room)
 1 Facing Open Doorway (when an open doorway is in front of the robot)
 2 In doorway (when passing through a doorway (not sure about this one)
 3 Facing Wall (seeing nothing but a wall)
 4 Hallway, Straight (oriented straight down a hallway)
 5 Hallway, Angled Right
 6 Hallway, Angled Left
 7 Open Space, like the atrium
 T Tile floor
 C Carpeting

 It is very similar to the locsForFrames program, except that the choices are more simple
"""

import time
import os
import cv2


class ClassifyFrames(object):
    def __init__(self, imageFolder):
        """Set up data to be held, including displayed and stored maps, current labeling location
        and heading, and dictionary mapping frame numbers to locations and headings."""


        # miscellaneous variables
        # only used if mode = video
        self.imgFileList = []
        self.picNum = -1
        self.dataSource = imageFolder
        self.dataDone = False

        # instance variables to hold displayed images
        self.currImage = None

        # Instance variables to hold outcome data
        self.classifications = dict()
        self.currEnvType = None
        self.currFloorType = None
        self.envToStr = {0: "In Room", 1: "Facing Door", 2: "In Doorway", 3: "Facing Wall",
                         4: "Hall, Straight", 5: "Hall, Angled Right", 6: "Hall, Angled Left", 7: "Open Space",
                         None: "None"}
        self.floorToStr = {'t': 'Tiled', 'c': 'Carpet', None: 'None'}


    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()


        self._setupImageCapture()
        self._printInstructions()
        ch = ' '

        i = 0
        # run main loop until user quits or run out of frames

        freshVisit = True
        while i < len(self.imgFileList) and (ch != 'q'):
            self._getNextImage(i, freshVisit)
            self._annotateAndDisplay(freshVisit)
            x = cv2.waitKey()
            ch = chr(x & 0xFF).lower()
            if ch == 't':
                self.currFloorType = 't'
                freshVisit = False
            elif ch == 'c':
                self.currFloorType = 'c'
                freshVisit = False
            elif ch in {'0', '1', '2', '3', '4', '5', '6', '7'}:
                self.currEnvType = int(ch)
                freshVisit = False
            elif ch == 'b':
                i -= 1
                freshVisit = True
            elif (ch == ' ') and (self.currEnvType is not None) and (self.currFloorType is not None):
                self.classifications[self.picNum] = [self.currEnvType, self.currFloorType]
                i += 1
                freshVisit = True

        # close things down nicely

        self._writeData()
        cv2.destroyAllWindows()


    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 350, 50)


    def _setupImageCapture(self):
        """Sets up the video capture for the given filename, printing a message if it failed
        and returning None in that case."""
        self.imgFileList = os.listdir(self.dataSource)
        self.imgFileList.sort()


    def _printInstructions(self):
        print "Must select Env Type and Floor Type prior to hitting the space key"
        print "Choose an environment type (Env Type) from the following:"
        for i in range(8):
            print i, "--", self.envToStr[i]
        print "Choose a floor type (Floor) from the following:"
        for key in ['t', 'c']:
            print key, '--', self.floorToStr[key]
        print
        print "Hit space to save this classification"
        print 'Hit q to quit'



    def _annotateAndDisplay(self, freshVisit):
        dispIm = self.currImage.copy()
        numStr = "Pic = " + str(self.picNum)
        if self.picNum in self.classifications and freshVisit:
            [envType, floorType] = self.classifications.get(self.picNum)
        else:
            [envType, floorType] = [self.currEnvType, self.currFloorType]
        envStr = "Env Type = " + self.envToStr.get(envType, None)
        floorStr = "Floor = " + self.floorToStr.get(floorType, None)

        cv2.putText(dispIm, numStr, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(dispIm, envStr, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(dispIm, floorStr, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.imshow("Image", dispIm)


    def _writeData(self):
        """Write the data collected to a timestamped file."""
        try:
            os.makedirs('classifydata/')
        except:
            pass
        logName = time.strftime("Data-%b%d%a-%H%M%S.txt")
        print(logName)
        fileOpen = False
        logFile = None
        try:
            logFile = open("classifydata/" + logName, 'w')
            fileOpen = True
        except:
            print("FAILED TO OPEN DATA FILE")

        dataTemplate = "{0:d} {1:d} {2:s}\n"
        for picNum in self.classifications:
            [envType, floorType] = self.classifications[picNum]
            dataStr = dataTemplate.format(picNum, envType, floorType)
            if fileOpen:
                logFile.write(dataStr)
            else:
                print("Image", picNum, "with classification", envType, floorType)
        logFile.close()



    def _getNextImage(self, index, freshVisit):
        """Gets the next frame from the camera or from reading the next file"""
        filename = self.imgFileList[index]
        thisNum = self._extractNum(filename)
        if freshVisit:
            newIm = cv2.imread(self.dataSource + filename)
            self.currImage = newIm
            self.picNum = thisNum
            if thisNum in self.classifications:
                [self.currEnvType, self.currFloorType] = self.classifications[thisNum]



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
    catkinPath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/"
    # catkinPath = "/home/macalester/"
    basePath = "catkin_ws/src/match_seeker/"

    frameClassifier = ClassifyFrames(catkinPath + basePath + "res/create060717/")
    frameClassifier.go()
