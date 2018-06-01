
""" ------------------------------------------------------------------------------------------------------------------
File: copyMarkedFiles.py
Authors: Susan Fox (building on code by Jane Pellegrini and JJ Lim)
Date: May 2018

Goes through a folder of pictures and its associated locations file, and displays each one to the user. If the user
types 'm' then the picture is moved to the new folder that is specified, and its entry in the locations data is copied to the new
locations file. Otherwise nothing is done

------------------------------------------------------------------------------------------------------------------"""

import os
from shutil import copy2
import cv2
import numpy as np
import readMap

class CopyMarkedFiles(object):


    def __init__(self, oldFramesPath, locationsFileName, newFramesPath, mapFile):
        self.oldFramesPath = oldFramesPath
        if self.oldFramesPath[-1] != "/":
            self.oldFramesPath += "/"
        self.oldLocationsFilename = locationsFileName
        self.newFramesPath = newFramesPath
        if self.oldFramesPath[-1] != "/":
            self.oldFramesPath += "/"
        self.newLocationsFilename = self._buildNewFilename()
        self.mapFile = mapFile
        self.toBeCopied = set()
        self.locData = self._readLocationData()
        self.filenames = self._getImageFilenames()
        self.getOlinMap()


    def queryCopy(self):
        self.setupWindows()
        i = 0
        filenames = self._getImageFilenames()

        while 0 <= i < len(self.filenames):
            currFile = self.filenames[i]

            currImg = cv2.imread(self.oldFramesPath + currFile)
            currImgNum = self._extractNum(currFile)
            currPose = self.locData[currImgNum]
            (currX, currY, currH) = currPose
            print(currImgNum, currPose)
            self.drawLocOnMap(currImgNum, currX, currY, currH)


            cv2.putText(currImg, str(currImgNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            cv2.imshow("Copy This?", currImg)

            x = cv2.waitKey(0)
            ch = chr(x & 0xFF)

            if (ch == 'q'):     # quit
                break
            elif (ch == 's'):  # save current data
                self.saveResults()
            elif (ch == 'c'):   # copy image to new directory
                self.toBeCopied.add(currFile)
                i += 1
            elif (ch == 'b'):   # "undo"
                if i > 0:
                    i -= 1
                if (self.filenames[i] in self.toBeCopied):
                    self.toBeCopied.remove(self.filenames[i])

            else:               # don't copy the image
                i += 1
        self.saveResults()


    def _buildNewFilename(self):
        """reworks the old filename to be a new filename with MOVED added just before the ."""
        parts = self.oldLocationsFilename.split('.')
        extension = parts[-1]
        parts.pop()  # remove extension part
        name = "".join(parts) + "MOVED" + "." + extension
        return name


    def _getImageFilenames(self):
        """Read filenames in folder, and keep those that end with jpg or png"""
        filenames = os.listdir(self.oldFramesPath)
        filenames.sort()
        keepers = []
        for name in filenames:
            if name.endswith("jpg") or name.endswith("png"):
                keepers.append(name)
        return keepers


    def _readLocationData(self):
        """Reads in the location file, building a dictionary that has the image number as key, and the location
        as its value."""
        locDict = dict()
        locFile = open(self.oldLocationsFilename, 'r')
        for line in locFile:
            parts = line.split()
            imageNum = int(parts[0])
            pose = [float(x) for x in parts[1:]]
            locDict[imageNum] = pose
        locFile.close()
        return locDict

    def saveResults(self):
        """Copy the images and save the locations data that corresponds."""
        self._copyImages()
        self._writeLocationsData()


    def _copyImages(self):
        """Copies each image in the toBeCopied set into the new folder"""
        if not os.path.isdir(self.newFramesPath):
            os.mkdir(self.newFramesPath)
        for imName in self.toBeCopied:
            imFullName = self.oldFramesPath + imName
            copy2(imFullName, self.newFramesPath)

    def _writeLocationsData(self):
        """Operating on the "to be moved" data, this writes the locations and image numbers to a file"""
        movedFiles = list(self.toBeCopied)
        movedFiles.sort()
        outFile = open(self.newLocationsFilename, 'w')
        for fName in movedFiles:
            imageNum = self._extractNum(fName)
            poseData = self.locData[imageNum]
            outFile.write(self._dataToString(imageNum, poseData))
        outFile.close()


    def _dataToString(self, imgNum, pose):
        lineTemplate = "{0:d} {1:f} {2:f} {3:f}\n"
        return lineTemplate.format(imgNum, pose[0], pose[1], pose[2])



    def _extractNum(self, fileString):
        """Finds sequence of digits"""
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


    def drawLocOnMap(self, nextNum, currX, currY, currH, offsetX=0, offsetY=0, offsetH=0):
        """Draws the current location on the map"""
        positionTemplate = "{0:d}: ({1:5.2f}, {2:5.2f}, {3:5.2f})"
        # offsetTemplate = "Offsets: ({0:5.2f}, {1:5.2f}, {2:d})"
        nextMapImg = self.olinMap.copy()
        (pixX, pixY) = self.convertWorldToMap(currX, currY)
        self.drawPosition(nextMapImg, pixX, pixY, currH, (255, 0, 0))
        posInfo = positionTemplate.format(nextNum, currX, currY, currH)
        # offsInfo = offsetTemplate.format(offsetX, offsetY, offsetH)
        cv2.putText(nextMapImg, posInfo, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(nextMapImg, offsInfo, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
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




if __name__ == "__main__":
    matchSeekerPath ="/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/"
    copier = CopyMarkedFiles(oldFramesPath=matchSeekerPath + "res/kobuki0615/",
                             newFramesPath=matchSeekerPath + "res/goodFrom2017/",
                             locationsFileName=matchSeekerPath + "res/locations/Susankobuki0615.txt",
                             mapFile=matchSeekerPath + "res/map/olinNewMap.txt" )

    copier.queryCopy()
