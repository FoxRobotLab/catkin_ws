#!/usr/bin/env python

""" ========================================================================
ImageRecognizer.py

Created: March 24, 2016
Author: Susan

This program extracts features from images and compares the images for
similarity based on those features. See the README for more details.

This is porting the CompareInfo class written in C++ in about 2011.
======================================================================== """


import sys
import os
import numpy as np
import readMap
import cv2
import OutputLogger
import ImageFeatures
from OSPathDefine import basePath
import MapGraph


class ImageMatcher(object):
    """..."""


    def __init__(self, bot, logFile = False, logShell = False,
                 dir1 = None, baseName = 'foo', ext= "jpg",
                 numMatches = 4):
        self.logToFile = logFile
        self.logToShell = logShell
        self.currDirectory = dir1
        self.baseName = baseName
        self.currExtension = ext
        self.numMatches = numMatches
        self.threshold = 800.0
        self.cameraNum = 0
        self.height = 0
        self.width = 0
        self.logger = OutputLogger.OutputLogger(self.logToFile, self.logToShell)

        self.robot = bot

        # Add line to debug ORB
        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()
        self.featureCollection = {} # dict with key being image number and value being ImageFeatures

        self.location = {}
        file = open(basePath + "scripts/buildingDatabases/locationsMay25.txt")
        for line in file.readlines():
            line = line.rstrip('/n')
            line = line.split()
            self.location[int(line[0])] = line[1:]

        self.path = basePath + "scripts/olinGraph.txt"
        self.olin = MapGraph.readMapFile(self.path)
        self.img = self.getOlinMap()


    def setLogToFile(self, val):
        if val in {False, True}:
            self.logToFile = val


    def setLogToShell(self, val):
        if val in {False, True}:
            self.logToShell = val


    def setCurrentDir(self, currDir):
        if type(currDir) == str:
            self.currDirectory = currDir


    def setSecondDir(self, sndDir):
        if type(sndDir) == str:
            self.secondDirectory = sndDir


    def setExtension(self, newExt):
        if type(newExt) == str:
            self.currExtension = newExt


    # def setFirstPicture(self, picNum):
    #     if type(picNum == int) and (picNum >= 0):
    #         self.startPicture = picNum

    # def setNumPictures(self, picNum):
    #     if type(picNum == int) and (picNum >= 0):
    #         self.numPictures = picNum

    def setCameraNum(self, camNum):
        if type(camNum == int) and (camNum >= 0):
            self.cameraNum = camNum

    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, creates a list of ImageFeatures objects

    def makeCollection(self):
        """Reads in all the images in the specified directory, start number and end number, and
        makes a list of ImageFeature objects for each image read in."""
        if (self.currDirectory is None):
            print("ERROR: cannot run makeCollection without a directory")
            return

        listDir = os.listdir(self.currDirectory)

        for file in listDir:
            image = cv2.imread(self.currDirectory + file)
            end = len(file) - (len(self.currExtension) + 1)
            picNum = int(file[len(self.baseName):end])
            if self.height == 0:
                self.height, self.width, depth = image.shape
            #self.logger.log("Image = " + str(picNum))
            features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
            self.featureCollection[picNum] = features
            # if i % 100 == 0:
            #     print i

        self.logger.log("Length of collection = " + str(len(self.featureCollection)))


    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, comparing all pairs in a range
    def matchImage(self, camImage):
        if len(self.featureCollection) == 0:
            print("ERROR: must have built collection before running this.")
            return
        # self.logger.log("Choosing frames from video to compare to collection")

        #grayCamImage = cv2.cvtColor(camImage, cv2.COLOR_BGR2GRAY)
        features = ImageFeatures.ImageFeatures(camImage, 9999, self.logger, self.ORBFinder)
        # cv2.imshow("Primary image", camImage)
        # cv2.moveWindow("Primary image", 0, 0)
        # features.displayFeaturePics("Primary image features", 0, 0)
        self._findBestNMatches(features, self.numMatches)


    def _findBestNMatches(self, features, numMatches):
        """Looks through the collection of features and keeps the numMatches
        best matches."""
        bestMatches = []
        bestScores = []
        for pos in self.featureCollection:
            feat = self.featureCollection[pos]
            simValue = features.evaluateSimilarity(feat)
            if simValue < self.threshold:
                if len(bestScores) < numMatches:
                    #self.logger.log("Adding good match " + str(len(bestScores)))
                    bestMatches.append(feat)
                    bestScores.append(simValue)
                elif len(bestScores) == numMatches:
                    whichMax = -1
                    maxBest = -1.0
                    #logStr = "Current scores = "
                    for j in range(numMatches):
                        #logStr += str(bestScores[j]) + " "
                        if bestScores[j] > maxBest:
                            maxBest = bestScores[j]
                            whichMax = j
                    #self.logger.log(logStr)
                    #self.logger.log("Current max best = " + str(maxBest) +
                                    #"    Current simValue = " + str(simValue))
                    if simValue < maxBest:
                        #self.logger.log("Changing " + str(whichMax) + "to new value")
                        bestScores[whichMax] = simValue
                        bestMatches[whichMax] = feat
                else:
                    self.logger.log("Should never get here... too many items in bestMatches!! " + str(len(bestMatches)))

        bestZipped = zip(bestScores, bestMatches)
        bestZipped.sort(cmp = lambda a, b: int(a[0] - b[0]))
        self.logger.log("==========Close Matches==========")
        if bestZipped[-1][0] > 99:
            self.logger.log("There are no good enough matches.")
        else:
            cv2.imshow("Match Picture", bestZipped[0][1].getImage())
            cv2.moveWindow("Match Picture", self.width + 10, 0)
            cv2.waitKey(20)
            img = self.img.copy()
            (self.mapHgt, self.mapWid, dep) = img.shape
            cv2.imshow("map",img)

            for j in range(len(bestZipped)-1, -1, -1):
                (nextScore, nextMatch) = bestZipped[j]
                # nextMatch.displayFeaturePics("Match Picture Features", self.width+10, 0)
                idNum = nextMatch.getIdNum()
                locX, locY, locHead = self.location[idNum]
                # self.logger.log("Image " + str(idNum) + " matches with similarity = " + str(nextScore))
                # print "x axis is", self.location[idNum][0], '. y axis is', self.location[idNum][1], '. Angle is', self.location[idNum][2], '.'
                (num, x, y) = self.findClosestNode((float(locX),float(locY)))
                print "The closest node is number", num, "Score:", nextScore
                pixelX,pixelY = self._convertWorldToMap(x,y)
                self.drawPosition(img, pixelX, pixelY, int(locHead),(0,0,255))
                turtleX, turtleY = self._convertWorldToMap(float(locX), float(locY))
                self.drawPosition(img, turtleX, turtleY,int(locHead),(255,nextScore*2.55,0))
                cv2.imshow("map",img)
                cv2.waitKey(20)

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
        cv2.line(image, (x,y), (newX, newY), color)


    def findClosestNode(self, (x, y)):
        closestNode = None
        closestX = None
        closestY = None
        for nodeNum in self.olin.getVertices():
            if closestNode is None:
                closestNode = nodeNum
                closestX, closestY = self.olin.getData(nodeNum)
                bestVal = ((closestX - x) * (closestX - x)) + ((closestY - y) * (closestY - y))
            (nodeX,nodeY) = self.olin.getData(nodeNum)
            val = ((nodeX - x) * (nodeX - x)) + ((nodeY - y) * (nodeY - y))
            if (val <= bestVal):
                bestVal = val
                closestNode = nodeNum
                closestX, closestY = (nodeX,nodeY)
        return (closestNode, closestX, closestY)

    def getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(basePath + "scripts/markLocations/olinNewMap.txt", 20)
        map2 = np.flipud(origMap)
        olinMap = np.rot90(map2)
        return olinMap

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

    # def _userGetInteger(self):
    #     """Ask until user either enters 'q' or a valid nonnegative integer"""
    #     inpStr = ""
    #     while not inpStr.isdigit():
    #         inpStr = raw_input("Enter nonnegative integer: ")
    #         if inpStr == 'q':
    #            return 'q'
    #     num = int(inpStr)
    #     return num


    def getFileByNumber(self, fileNum):
        """Makes a filename given the number and reads in the file, returning it."""
        filename = self.makeFilename(fileNum)
        image = cv2.imread(filename)
        if image is None:
            print("Failed to read image:", filename)
        return image


    def putFileByNumber(self, fileNum, image):
        """Writes a file in the current directory with the given number."""
        filename = self.makeFilename(fileNum)
        cv2.imwrite(filename, image)


    def makeFilename(self, fileNum):
        """Makes a filename for reading or writing image files"""
        formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
        name = formStr.format(self.currDirectory,
                              self.baseName,
                              fileNum,
                              self.currExtension)
        return name




    # def run(self):
    #     runFlag = True
    #     self.makeCollection()
    #     if len(self.featureCollection) == 0:
    #         print("ERROR: must have built collection before running this.")
    #         return
    #     self.logger.log("Choosing frames from video to compare to collection")
    #     print("How many matches should it find?")
    #     numMatches = self._userGetInteger()
    #     while runFlag:
    #         image = self.robot.getImage()[0]
    #         if image is None:
    #             break
    #         features = ImageFeatures.ImageFeatures(image, 9999, self.logger, self.ORBFinder)
    #         cv2.imshow("Primary image", image)
    #         cv2.moveWindow("Primary image", 0, 0)
    #         features.displayFeaturePics("Primary image features", 0, 0)
    #         self._findBestNMatches(features, numMatches)
    #         runFlag = self.runFlag

    # def isStalled(self):
    #     """Returns the status of the camera stream"""
    #     return self.stalled
    #
    #
    # def exit(self):
    #     self.runFlag = False


if __name__ == '__main__':

    matcher = ImageMatcher(logFile = True, logShell = True,
                           dir1 = basePath + "res/May2517/",
                           baseName = "frame",
                           ext = "jpg")




