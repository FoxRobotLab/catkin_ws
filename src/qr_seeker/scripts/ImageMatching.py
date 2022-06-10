#!/usr/bin/env python

""" ========================================================================
ImageMatching.py

Created: March 24, 2016
Author: Susan

This program extracts features from images and compares the images for
similarity based on those features. See the README for more details.

This is porting the CompareInfo class written in C++ in about 2011.
======================================================================== """


import sys
import rospy
import cv2
import OutputLogger
import ImageFeatures
import os
import turtleQR
import MapGraph
import readMap
import numpy as np

basePath = "/home/macalester/catkin_ws/src/match_seeker/"
directory = "res/BWmay25Dir/"
locData = "scripts/buildingDatabases/locationsMay25ORIG.txt"

class ImageMatcher(object):
    """..."""


    def __init__(self, logFile = False, logShell = False,
                 dir1 = None, dir2 = None,
                 baseName = 'foo', ext= "jpg",
                 startPic = 0, numPics = -1):
        self.logToFile = logFile
        self.logToShell = logShell
        self.currDirectory = dir1
        self.secondDirectory = dir2
        self.baseName = baseName
        self.currExtension = ext
        self.startPicture = startPic
        self.numPictures = numPics
        self.threshold = 800.0
        self.cameraNum = 0
        self.height = 0
        self.width = 0
        self.stalled = False
        self.runFlag = True
        print "before init outputLogger"
        self.logger = OutputLogger.OutputLogger(self.logToFile, self.logToShell)
        print "made logger"

        # self.robot = turtleQR.TurtleBot()


        # Add line to debug ORB
        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()
        self.featureCollection = {} # dict with key being image number and value being ImageFeatures

        self.location = {}
        file = open(basePath + locData)
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


    def setFirstPicture(self, picNum):
        if type(picNum == int) and (picNum >= 0):
            self.startPicture = picNum

    def setNumPictures(self, picNum):
        if type(picNum == int) and (picNum >= 0):
            self.numPictures = picNum

    def setCameraNum(self, camNum):
        if type(camNum == int) and (camNum >= 0):
            self.cameraNum = camNum


    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, comparing all pairs in a range

    def cycle(self, verbose = False):
        """Compares all to all. Must have called makeCollection before calling this operation."""

        if (self.currDirectory is None)\
           or (self.numPictures == -1):
            print("ERROR: cannot run cycle without at least a directory and a number of pictures")
            return
        elif len(self.featureCollection) == 0:
            print("ERROR: must have collection built before cycling")
            return
        self.logger.log("Comparing all pictures with all")

        quitTime = False
        matchScore = {}
        #cv2.namedWindow("Image l")
        #cv2.namedWindow("Image 2")

        for i in range(self.numPictures):
            picNum = self.startPicture + i
            if quitTime:
                break
            features1 = self.featureCollection[picNum] #_getImageAndDisplay(picNum, "Image 1", 100, 0)
            if verbose:
                features1.displayFeaturePics("Image 1", 0, 0)
                self.logger.log("image 1 = " + str(picNum))

            for j in range(self.numPictures):  # Note this duplicates for debugging purposes comparisons both ways
                pic2num = self.startPicture + j
                features2 = self.featureCollection[pic2num] #self._getImageAndDisplay(pic2num, "Image 2", 400, 0)
                if verbose:
                    features2.displayFeaturePics("Image 2", self.width+10, 0)
                simVal = features1.evaluateSimilarity(features2)
                matchScore[picNum, pic2num] = simVal
                logStr = "image " + str(picNum) + " matched image " + str(pic2num) + " with score " + str(simVal)
                self.logger.log(logStr)

                c = cv2.waitKey(50)
                #c = chr(c & 0xFF)
                #if c == 'q':
                    #quitTime = True
                #elif c == 'm':
                    #self.threshold += 50
                #elif c == 'l':
                    #self.threshold -= 50

        # Displays to logger a table of the pictures and their similarity scores
        formatSimilScore = "{:^9.3f}"
        formatPicNum = "  {:0>4d}   "
        firstLine = " " * 9
        for i in range(self.numPictures):
            pNum = self.startPicture + i
            nextStr = formatPicNum.format(pNum)
            firstLine += nextStr
        self.logger.log(firstLine)
        for i in range(self.numPictures):
            pNum1 = self.startPicture + i
            nextLine = formatPicNum.format(pNum1)
            for j in range(self.numPictures):
                pNum2 = self.startPicture + j
                nextStr = formatSimilScore.format(matchScore[pNum1, pNum2])
                nextLine += nextStr
            self.logger.log(nextLine)



    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, user-selected pairs of pictures ...

    def compareSelected(self):
        """Loops until the user says to quit. It asks the user for two image numbers,
        reads those images from the current directory, and compares them, displaying the
        results."""
        if (self.currDirectory is None):
            print("ERROR: cannot run compareSelected without at least a directory")
            return
        self.logger.log("Comparing user-selected pictures...")

        cv2.namedWindow("Image l")
        cv2.namedWindow("Image 2")


        while True:
            print("Enter the number of the first picture")
            pic1Num = self._userGetInteger()
            if pic1Num == 'q':
                break
            print("Enter the number of the second picture")
            pic2Num = self._userGetInteger()
            if pic2Num == 'q':
                break

            image1, features1 = self._getImageAndDisplay(pic1Num, "Image 1", 0, 0)
            if self.height == 0:
                self.height, self.width, depth = image1.shape
            self.logger.log("image 1 =" + str(pic1Num))
            image2, features2 = self._getImageAndDisplay(pic2Num, "Image 2", self.width+10, 0)
            self.logger.log("image 2 =" + str(pic2Num))
            simval = features1.evaluateSimilarity(features2)
            self.logger.log("Similarity = " + str(simval))



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
            pic = cv2.imread(self.currDirectory + file)
            end = len(file) - (len(self.currExtension) + 1)
            picNum = int(file[len(self.baseName):end])
            features = ImageFeatures.ImageFeatures(pic, picNum, self.logger, self.ORBFinder)
            self.featureCollection[picNum] = features


    # def makeCollection(self):
    #     """Reads in all the images in the specified directory, start number and end number, and
    #     makes a list of ImageFeature objects for each image read in."""
    #     print "in make collection"
    #     if (self.currDirectory is None)\
    #        or (self.numPictures == -1):
    #         print("ERROR: cannot run makeCollection without a directory and a number of pictures")
    #         return
    #     self.logger.log("Reading in image database")
    #
    #
    #     for i in range(self.numPictures):
    #         picNum = self.startPicture + i
    #         image = self.getFileByNumber(picNum)
    #         if self.height == 0:
    #             self.height, self.width, depth = image.shape
    #         #self.logger.log("Image = " + str(picNum))
    #         features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
    #         self.featureCollection[picNum] = features
    #         print i
    #
    #     self.logger.log("Length of collection = " + str(self.numPictures))


    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, creates a list of ImageFeatures objects
    def mostSimilarSelected(self):
        """user selects images, and how many matches, and this finds and displas the  most similar."""
        if (self.currDirectory is None)\
           or (self.numPictures == -1):
            print("ERROR: cannot run mostSimilarSelected without a directory and a number of pictures")
            return
        elif len(self.featureCollection) == 0:
            print("ERROR: must have built collection before running this.")
            return
        self.logger.log("Finding most similar to selected")

        quitTime = False
        print("How many matches should it find?")
        numMatches = self._userGetInteger()
        while not quitTime:
            print("Enter the number of the primary picture.")
            picNum = self._userGetInteger()
            if picNum == 'q':
                break
            image, features = self._getImageAndDisplay(picNum, "Primary image", 0, 0)
            self.logger.log("Primary image = " + str(picNum))

            self._findBestNMatches(image, features, numMatches)


    # ------------------------------------------------------------------------
    # # One of the major operations we can undertake, comparing all pairs in a range
    # def mostSimilarCamera(self):
    #     """Connects to the camera and when user hits a key it takes that picture and compares against the collection."""
    #     if len(self.featureCollection) == 0:
    #         print("ERROR: must have built collection before running this.")
    #         return
    #     self.logger.log("Choosing frames from video to compare to collection")
    #     print("How many matches should it find?")
    #     numMatches = self._userGetInteger()
    #     quitTime = False
    #     # cap = cv2.VideoCapture(self.cameraNum)
    #     while not quitTime:
    #         # image = self._userSelectFrame(cap)
    #         # image = self.robot.getImage()[0]
    #         if image is None:
    #             break
    #         features = ImageFeatures.ImageFeatures(image, 9999, self.logger, self.ORBFinder)
    #         cv2.imshow("Primary image", image)
    #         cv2.moveWindow("Primary image", 0, 0)
    #         features.displayFeaturePics("Primary image features", 0, 0)
    #         self._findBestNMatches(image, features, numMatches)



    def _userSelectFrame(self, cap):
        """User types 'q' to quit without selecting, or space to
        select the current frames. Frame or None is returned."""
        while True:
            r, frame = cap.read()
            if not r:
                return None
            cv2.imshow("Webcam View", frame)
            inp = cv2.waitKey(30)
            ch = chr(inp % 255)
            if ch == 'q':
                return None
            elif ch == ' ':
                return frame





    def _findBestNMatches(self, image, features, numMatches):
        """Looks through the collection of features and keeps the numMatches
        best matches."""
        bestMatches = []
        bestScores = []
        for pos in self.featureCollection:
            print pos
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

        # for j in range(len(bestZipped)):
        #     (nextScore, nextMatch) = bestZipped[j]
        #     cv2.imshow("Match Picture", nextMatch.getImage())
        #     cv2.moveWindow("Match Picture", self.width+10, 0)
        #     nextMatch.displayFeaturePics("Match Picture Features", self.width+10, 0)
        #     self.logger.log("Image " + str(nextMatch.getIdNum()) + " matches with similarity = " + str(nextScore))
        #     cv2.waitKey(0)


        self.img = self.img.copy()
        (self.mapHgt, self.mapWid, dep) = self.img.shape
        # cv2.imshow("map", img)
        # cv2.waitKey(0)

        for j in range(len(bestZipped)):
            (nextScore, nextMatch) = bestZipped[j]
            # nextMatch.displayFeaturePics("Match Picture Features", self.width+10, 0)
            idNum = nextMatch.getIdNum()
            # self.logger.log("Image " + str(idNum) + " matches with similarity = " + str(nextScore))
            # print "x axis is", self.location[idNum][0], '. y axis is', self.location[idNum][1], '. Angle is', self.location[idNum][2], '.'
            (num, x, y) = self.findClosestNode((float(self.location[idNum][0]), float(self.location[idNum][1])))
            # print "The closest node is number", num, "with x and y coordinates as", x, "and", y
            pixelX, pixelY = self._convertWorldToMap(x, y)

            cv2.circle(self.img, (pixelX,pixelY),6, 255)
            cv2.imshow("map", self.img)
            cv2.waitKey(20)

    def findClosestNode(self, (x, y)):
        closestNode = None
        closestX = None
        closestY = None
        for nodeNum in self.olin.getVertices():
            if closestNode is None:
                closestNode = nodeNum
                closestX, closestY = self.olin.getData(nodeNum)
                bestVal = ((closestX - x) * (closestX - x)) + ((closestY - y) * (closestY - y))
            (nodeX, nodeY) = self.olin.getData(nodeNum)
            val = ((nodeX - x) * (nodeX - x)) + ((nodeY - y) * (nodeY - y))
            if (val <= bestVal):
                bestVal = val
                closestNode = nodeNum
                closestX, closestY = (nodeX, nodeY)
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



    def _userGetInteger(self):
        """Ask until user either enters 'q' or a valid nonnegative integer"""
        inpStr = ""
        while not inpStr.isdigit():
            inpStr = raw_input("Enter nonnegative integer: ")
            if inpStr == 'q':
               return 'q'
        num = int(inpStr)
        return num


    def _getImageAndDisplay(self, picNum, label, baseX, baseY):
        """Given a picture number and a label, it reads the image, creates
        its features, and displays the features, returning the image and its
        features as the value."""
        image = self.getFileByNumber(picNum)
        cv2.imshow(label, image)
        cv2.moveWindow(label, baseX, baseY)
        cv2.waitKey(50)
        features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
        features.displayFeaturePics(label + "Features", baseX, baseY)
        return image, features



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
    #     # cap = cv2.VideoCapture(self.cameraNum)
    #     while runFlag:
    #         # image = self._userSelectFrame(cap)
    #         # image = self.robot.getImage()[0]
    #         if image is None:
    #             break
    #         features = ImageFeatures.ImageFeatures(image, 9999, self.logger, self.ORBFinder)
    #         cv2.imshow("Primary image", image)
    #         cv2.moveWindow("Primary image", 0, 0)
    #         features.displayFeaturePics("Primary image features", 0, 0)
    #         self._findBestNMatches(image, features, numMatches)
    #         runFlag = self.runFlag


    def isStalled(self):
        """Returns the status of the camera stream"""
        return self.stalled


    def exit(self):
        self.runFlag = False


if __name__ == '__main__':
    # rospy.init_node('ImageMatching')
    print "GOT HERE"
    matcher = ImageMatcher(logFile = True, logShell = True,
                           dir1 = basePath + directory,
                           baseName = "frames",
                           ext = "jpg", startPic = 0,
                           numPics = 1086)
    print "finish ImageMatcher call"
    matcher.makeCollection()
    matcher.mostSimilarSelected()

    # matcher.run()
    # rospy.on_shutdown(matcher.exit)
    # #matcher.mostSimilarSelected()
    # rospy.spin()



