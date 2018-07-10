""" ------------------------------------------------------------------------------------------------------------------
File: checkLocs.py
Authors: Susan Fox, Xinyu Yang, and Kayla Beckham
Date: May/June 2017

for double checking your existing location tags. YOU SHOULD RUN CHECKLOCS EVERY TIME YOU TAG LOCATIONS.
It has almost the same controls as locsForFrames, but uses X/C to change angles. Also, checkLocs keeps up with an
ongoing offset, so sometimes after you move one node forward a few feet, you may need to move the next one back.

Takes in a locations file that maps pictures to locations, and the folder of pictures that corresponds with it, and
it draws the location of each picture on a map of Olin-Rice. It then allows the user to move the picture to a new location
and/or heading, and it writes the new data, changed or unchanged, to a new location file. The new file just has the label
NEW tacked on to the filename for the old location file.
------------------------------------------------------------------------------------------------------------------"""

import os
import math

import cv2

from DataPaths import basePath, imageDirectory, locData
from OlinWorldMap import WorldMap


class CheckerOfLocs(object):
    """An object that can read locations and pictures and interactively asks the user to correct the locations and
    headings."""

    def __init__(self, locFile, imageDir, saveAll = True, nearLoc = None, onlyNear = False, startLoc = 0):
        """Sets up the location file and image folder..."""

        self.locFile = locFile
        self.imageDir = imageDir
        self.saveAll = saveAll
        self.nearLoc = nearLoc
        self.onlyNear = onlyNear
        self.startingLoc = startLoc

        self.oldLocations = dict()
        self.newLocations = dict()
        self.olinMap = WorldMap()
        (self.mapHgt, self.mapWid, self.mapDep) = (None, None, None)


    def go(self):
        """Actually processes the data, and saves it to a file"""
        self.readLocations()
        self.displayAndProcess()
        self.saveNewLocations()


    def displayAndProcess(self):
        """Displays the images, and map, drawing the location on the map. User interactively updates the location."""
        self.setupWindows()
        self.olinMap.cleanMapImage(cells=True)
        self.olinMap.displayMap("Map")

        filenames = self.getImageFilenames()

        i = self.startingLoc
        offsetX = 0
        offsetY = 0
        offsetH = 0

        while i < len(filenames):
            name = filenames[i]
            nextNum = self.extractNum(name)
            if nextNum < 0:
                print "Weird filename in folder: ", name, "... skipping!"
                continue

            (locX, locY, locH) = self.getCurrLocInfo(nextNum)
            if self.onlyNear and not self.closeEnough(locX, locY):
                i = i+1
                continue
            currX = locX + offsetX
            currY = locY + offsetY
            currH = (locH + offsetH) % 360

            self.olinMap.cleanMapImage(cells=True)
            self.olinMap.drawPose((currX, currY, currH))
            self.olinMap.displayMap("Map")

            nextImg = cv2.imread(self.imageDir + name)
            self.drawLocOnImage(nextImg, nextNum, (currX, currY, currH), (offsetX, offsetY, offsetH))
            cv2.imshow("Image", nextImg)

            res = cv2.waitKey()
            userKey = chr(res % 0xFF)
            if userKey == 'q':
                break
            elif userKey == ' ':
                if self.saveAll or (offsetH > 0) or (offsetX != 0.0) or (offsetY != 0.0):
                    self.newLocations[nextNum] = [currX, currY, currH]
                i += 1
                offsetH = 0
            elif userKey == 'b':
                i -= 1
                offsetX = 0
                offsetY = 0
                offsetH = 0
            elif userKey == 'w':
                offsetX += 0.1
            elif userKey == 's':
                offsetX -= 0.1
            elif userKey == 'a':
                offsetY += 0.1
            elif userKey == 'd':
                offsetY -= 0.1
            elif userKey == 'z':
                offsetH += 45
                if offsetH >= 360:
                    offsetH -= 360
            elif userKey == 'c':
                offsetH -= 45
                if offsetH < 0:
                    offsetH += 360

    def getImageFilenames(self):
        """Get the image filenames using os module"""
        # get filenames of image files
        filenames = os.listdir(self.imageDir)
        filenames.sort()
        keepers = []
        for name in filenames:
            if name.endswith('jpg') or name.endswith('png'):
                keepers.append(name)
        return keepers


    def readLocations(self,):
        """Read locations from a file and store in a dictionary."""
        locations = dict()
        f = open(self.locFile, 'r')
        for line in f.readlines():
            line = line.rstrip('/n')
            line = line.split()
            locations[int(line[0])] = [float(line[1]), float(line[2]), float(line[3])]
        f.close()
        self.oldLocations = locations



    def saveNewLocations(self):
        """Writes the data from self.newLocations to a new file"""
        lineTemplate = "{0:d} {1:.2f} {2:.2f} {3:.1f}\n"

        newLocFile = open(self.locFile + "NEW", 'w')
        newKeys = self.newLocations.keys()
        newKeys.sort()
        for locKey in newKeys:
            [currX, currY, currH] = self.newLocations[locKey]
            nextLine = lineTemplate.format(locKey, currX, currY, currH)
            newLocFile.write(nextLine)
        newLocFile.close()



    def setupWindows(self):
        """Creates three windows and moves them to the right places on screen."""
        cv2.namedWindow("Image")
        cv2.namedWindow("Map")
        cv2.moveWindow("Image", 30, 50)
        cv2.moveWindow("Map", 100, 50)


    def drawLocOnImage(self, nextImg, nextNum, currPose, offsetPose):
        """Draws the current location on the map"""
        currX, currY, currH = currPose
        offsetX, offsetY, offsetH = offsetPose
        positionTemplate = "{0:d}: ({1:5.2f}, {2:5.2f}, {3:.0f})"
        offsetTemplate = "Offsets: ({0:5.2f}, {1:5.2f}, {2:.0f})"
        cellTemplate = "Cell: {0:s}"
        posInfo = positionTemplate.format(nextNum, currX, currY, currH)
        offsInfo = offsetTemplate.format(offsetX, offsetY, offsetH)
        cellInfo = cellTemplate.format(self.olinMap.convertLocToCell((currPose)))
        cv2.putText(nextImg, posInfo, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(nextImg, offsInfo, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(nextImg, cellInfo, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)





    def extractNum(self, fileString):
        """finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for c in fileString:
            if c in '0123456789':
                foundDigits = True
                numStr += c
            elif foundDigits:
                break
        if numStr != "":
            return int(numStr)
        else:
            return -1

    def getCurrLocInfo(self, imgNum):
        if imgNum in self.newLocations:
            return self.newLocations[imgNum]
        else:
            if self.saveAll:
                self.newLocations[imgNum] = self.oldLocations[imgNum]
                return self.newLocations[imgNum]
            else:
                return self.oldLocations[imgNum]


    def closeEnough(self, locX, locY):
        """Checks if a given (x, y) location is close enough to the "near to" location. Note that there may
        be no "near to" location, in which case it returns True."""
        if self.nearLoc == None:
            return True
        else:
            dist = math.sqrt((self.nearLoc[0] - locX) ** 2 + (self.nearLoc[1] - locY) ** 2)
            return dist <= 8


if __name__ == '__main__':
    checker = CheckerOfLocs(basePath + locData,
                            basePath + imageDirectory,
                            saveAll = True,
                            startLoc=3725)
    checker.go()
