"""
File: classifyFrames.py
Author: Susan Fox
Date: February 2017

This file allows the user to select among a set of options for describing the environment the picture is in.
Categories:
Space type:
 0 In Room  (just a view of a random room)
 1 Hallway
 2 Open space (like atrium or right angle)

 Orientation to space:
 0 None (if no orientation makes sense, like in an open space or room)
 1 Straight (oriented in alignment with space or usual direction of movement)
 2 Angled left
 3 Angled right
 4 Perpendicular

 Visible Walls:
 0 None
 1 On left
 2 On right
 3 Both sides


 Facing:
 0 Open space
 1 Open doorway (like when entering/leaving lab, but also area by right angle, and what about hallway ends into atrium)
 2 Closed door
 3 Wall
 (4 Elevator (as special case?))

 Floor type:
 0 Carpet
 1 Tile

 It is very similar to the locsForFrames program, except that the choices are more simple
"""

import time
import os
import cv2
import numpy as np


class ClassifyFrames(object):
    def __init__(self, imageFolder):
        """Set up data to be held, including displayed and stored maps, current labeling location
        and heading, and dictionary mapping frames numbers to locations and headings."""
        self.categories = ['Space Type', 'Orientation', 'Side Walls', 'Facing', 'Floor Type']
        self.envMapToStr = {'Space Type': {0: 'in room', 1: 'hallway', 2: 'open space'},
                            'Orientation': {0: 'none', 1: 'straight', 2: 'angled left', 3: 'angled right',
                                            4: 'perpendicular'},
                            'Side Walls': {0: 'none', 1: 'on left', 2:  'on right', 3: 'both sides'},
                            'Facing': {0: 'open space', 1: 'open doorway', 2: 'closed door', 3: 'wall'},
                            'Floor Type': {0: 'carpet', 1: 'tile'} }
        self.catNumValues = dict()
        for cat in self.categories:
            self.catNumValues[cat] = len(self.envMapToStr[cat])

        # miscellaneous variables
        self.imgFileList = []
        self.picNum = -1
        self.dataSource = imageFolder
        self.dataDone = False

        # instance variables to hold displayed images
        self.currImage = None

        # Instance variables to hold outcome data
        self.classifications = dict()
        self.currEnvironment = {'Space Type': 0,
                                'Orientation': 0,
                                'Side Walls': 0,
                                'Facing': 0,
                                'Floor Type': 0}


    def go(self):
        """Run the program, setting up windows and all."""
        self._setupWindows()


        self._setupImageList()
        self._printInstructions()
        ch = ' '

        i = 0
        # run main loop until user quits or run out of frames

        freshVisit = True
        while 0 <= i < len(self.imgFileList):
            self._getNextImage(i, freshVisit)
            self._displayEnvironment(freshVisit)

            cv2.imshow("Image", self.currImage)
            x = cv2.waitKey()
            ch = chr(x & 0xFF).lower()
            if ch == 'q':
                break
            elif ch in {'1', '2', '3', '4', '5'}:  # Update corresponding feature
                featIndex = int(ch) - 1
                self._incrementEnvFeature(self.categories[featIndex])
                freshVisit = False
            elif ch == 'b':
                i -= 1
                freshVisit = True
            elif ch == ' ':
                self.classifications[self.picNum] = self.currEnvironment.copy()
                i += 1
                freshVisit = True

        # close things down nicely

        self._writeData()
        cv2.destroyAllWindows()


    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 450, 50)
        cv2.namedWindow("Status")
        cv2.moveWindow('Status', 50, 50)


    def _setupImageList(self):
        """Sets up the video capture for the given filename, printing a message if it failed
        and returning None in that case."""
        folderFiles = os.listdir(self.dataSource)
        self.imgFileList = []
        for fName in folderFiles:
            if fName.endswith('jpg') or fName.endswith('png'):
                self.imgFileList.append(fName)
        self.imgFileList.sort()

    def _incrementEnvFeature(self, category):
        """Takes a category and updates its value to the next in sequence, cycling around to 0 when it
        hits the max value"""
        currVal = self.currEnvironment[category]
        limit = self.catNumValues[category]
        nextVal = (currVal + 1) % limit
        self.currEnvironment[category] = nextVal


    def _printInstructions(self):
        print "Hitting the number corresponding to each feature category will cycle through"
        print "the options for it."
        print "Hit space to save this classification and go on to the next picture"
        print 'Hit 'b' to go back to the previous picture, restoring the old value if one was saved'
        print 'Hit q to quit'



    def _displayEnvironment(self, freshVisit):
        displayImg = np.ones((400, 400, 3), np.uint8)
        numStr = "Pic = " + str(self.picNum)

        cv2.putText(displayImg, numStr, (180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        yValue = 70
        yOffset = 50
        keyNum = 1
        for cat in self.categories:
            cv2.putText(displayImg, str(keyNum), (20, yValue), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(displayImg, cat, (60, yValue), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            val = self.currEnvironment[cat]
            valDescr = self.envMapToStr[cat][val]
            cv2.putText(displayImg, str(val), (200, yValue), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            cv2.putText(displayImg, valDescr, (230, yValue), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            yValue += yOffset
            keyNum += 1

        cv2.imshow("Status", displayImg)

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

        # dataTemplate = "{0:d} {1:d} {2:d} {3:d} {4:d} {5:d}\n"
        for picNum in self.classifications:
            classDict = self.classifications[picNum]
            dataStr = str(picNum) + ' '
            for cat in self.categories:
                nextVal = classDict[cat]
                dataStr += str(nextVal) + ' '
            dataStr += '\n'
            if fileOpen:
                logFile.write(dataStr)
            else:
                print("Data String: ", dataStr)
        logFile.close()



    def _getNextImage(self, index, freshVisit):
        """Gets the next frames from the camera or from reading the next file"""
        filename = self.imgFileList[index]
        thisNum = self._extractNum(filename)
        if freshVisit:
            newIm = cv2.imread(self.dataSource + filename)
            self.currImage = newIm
            self.picNum = thisNum
            if thisNum in self.classifications:
                self.currEnvironment = self.classifications[thisNum].copy()



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

    frameClassifier = ClassifyFrames(catkinPath + basePath + "res/allFrames060418/")
    frameClassifier.go()
