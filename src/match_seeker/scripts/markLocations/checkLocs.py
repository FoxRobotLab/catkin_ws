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
import numpy as np

import readMap


class CheckerOfLocs(object):
    """An object that can read locations and pictures and interactively asks the user to correct the locations and
    headings."""

    def __init__(self, mapFile, locFile, imageDir, saveAll = True, nearLoc = None, onlyNear = False):
        """Sets up the location file and image folder..."""
        self.mapFile = mapFile
        self.locFile = locFile
        self.imageDir = imageDir
        self.saveAll = saveAll
        self.nearLoc = nearLoc
        self.onlyNear = onlyNear

        self.oldLocations = dict()
        self.newLocations = dict()
        self.olinMap = None
        (self.mapHgt, self.mapWid, self.mapDep) = (None, None, None)


    def go(self):
        """Actually reads the data, processes it, and saves it to a file"""
        self.readLocations()
        self.displayAndProcess()
        self.saveNewLocations()


    def displayAndProcess(self):
        """Displays the images, and map, drawing the location on the map. User interactively updates the location."""
        self.getOlinMap()
        self.setupWindows()
        # get filenames of image files
        filenames = os.listdir(self.imageDir)
        filenames.sort()
        i = 0
        offsetX = 0
        offsetY = 0
        offsetH = 0

        while i < len(filenames):
            name = filenames[i]
            nextNum = self.extractNum(name)
            if nextNum < 0:
                print "Weird filename in folder: ", name, "... skipping!"
                continue

            nextImg = cv2.imread(self.imageDir + name)
            cv2.imshow("Image", nextImg)
            (locX, locY, locH) = self.getCurrLocInfo(nextNum)
            if self.onlyNear and not self.closeEnough(locX, locY):
                i = i+1
                continue
            currX = locX + offsetX
            currY = locY + offsetY
            currH = (locH + offsetH) % 360
            self.drawLocOnMap(nextNum, currX, currY, currH, offsetX, offsetY, offsetH)
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



    def readLocations(self,):
        """Read locations from a file and store in a dictionary."""
        locations = dict()
        f = open(self.locFile, 'r')
        for line in f.readlines():
            line = line.rstrip('/n')
            line = line.split()
            locations[int(line[0])] = [float(line[1]), float(line[2]), int(line[3])]
        f.close()
        self.oldLocations = locations



    def saveNewLocations(self):
        """Writes the data from self.newLocations to a new file"""
        lineTemplate = "{0:d} {1:.2f} {2:.2f} {3:d}\n"

        newLocFile = open(self.locFile + "NEW", 'w')
        newKeys = self.newLocations.keys()
        newKeys.sort()
        for locKey in newKeys:
            [currX, currY, currH] = self.newLocations[locKey]
            nextLine = lineTemplate.format(locKey, currX, currY, currH)
            newLocFile.write(nextLine)
        newLocFile.close()


    def getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(self.mapFile, 20)
        map2 = np.flipud(origMap)
        self.olinMap = np.rot90(map2)
        (self.mapHgt, self.mapWid, self.mapDep) = self.olinMap.shape


    def setupWindows(self):
        """Creates three windows and moves them to the right places on screen."""
        cv2.namedWindow("Image")
        cv2.namedWindow("Map")
        cv2.moveWindow("Image", 30, 50)
        cv2.moveWindow("Map", 700, 50)


    def drawLocOnMap(self, nextNum, currX, currY, currH, offsetX, offsetY, offsetH):
        """Draws the current location on the map"""
        positionTemplate = "{0:d}: ({1:5.2f}, {2:5.2f}, {3:d})"
        offsetTemplate = "Offsets: ({0:5.2f}, {1:5.2f}, {2:d})"
        nextMapImg = self.olinMap.copy()
        (pixX, pixY) = self.convertWorldToMap(currX, currY)
        self.drawPosition(nextMapImg, pixX, pixY, currH, (255, 0, 0))
        posInfo = positionTemplate.format(nextNum, currX, currY, currH)
        offsInfo = offsetTemplate.format(offsetX, offsetY, offsetH)
        cv2.putText(nextMapImg, posInfo, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(nextMapImg, offsInfo, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.imshow("Map", nextMapImg)


    def drawPosition(self, image, x, y, heading, color):
        cv2.circle(image, (x, y), 6, color, -1)
        newX = x
        newY = y
        if heading == 0:
            newY = y - 10
        elif heading == 45:
            newX = x - 8
            newY = y - 8
        elif heading == 90:
            newX = x - 10
        elif heading == 135:
            newX = x - 8
            newY = y + 8
        elif heading == 180:
            newY = y + 10
        elif heading == 225:
            newX = x + 8
            newY = y + 8
        elif heading == 270:
            newX = x + 10
        elif heading == 315:
            newX = x + 8
            newY = y - 8
        else:
            print "Error! The heading is", heading
        cv2.line(image, (x, y), (newX, newY), color)


    def convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return mapXMeters, mapYMeters


    def convertWorldToMap(self, worldX, worldY):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        pixelX = worldX * 20.0
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.mapWid - 1 - pixelY
        mapY = self.mapHgt - 1 - pixelX
        return int(mapX), int(mapY)


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
    catkinPath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/"
    basePath = "catkin_ws/src/match_seeker/"
    checker = CheckerOfLocs(catkinPath + basePath + "res/map/olinNewMap.txt",
                            catkinPath + basePath + "res/locations/kobuki0615.txt",
                            catkinPath + basePath + "res/kobuki0615/",
                            saveAll = True) # nearLoc = (7.5, 6.4), onlyNear = True)
    checker.go()
