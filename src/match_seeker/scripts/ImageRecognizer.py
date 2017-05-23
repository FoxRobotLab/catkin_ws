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
import rospy
import cv2
import OutputLogger
import ImageFeatures


class ImageMatcher(object):
    """..."""


    def __init__(self, bot, logFile = False, logShell = False,
                 dir1 = None, baseName = 'foo', ext= "jpg",
                 startPic = 0, numPics = -1, numMatches = 4):
        self.logToFile = logFile
        self.logToShell = logShell
        self.currDirectory = dir1
        self.baseName = baseName
        self.currExtension = ext
        self.startPicture = startPic
        self.numPictures = numPics
        self.numMatches = numMatches
        self.threshold = 800.0
        self.cameraNum = 0
        self.height = 0
        self.width = 0
        print "before init outputLogger"
        input("type something")
        self.logger = OutputLogger.OutputLogger(self.logToFile, self.logToShell)
        print "made logger"

        self.robot = bot

        # Add line to debug ORB
        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()
        self.featureCollection = {} # dict with key being image number and value being ImageFeatures


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
    # One of the major operations we can undertake, creates a list of ImageFeatures objects

    def makeCollection(self):
        """Reads in all the images in the specified directory, start number and end number, and
        makes a list of ImageFeature objects for each image read in."""
        print "in make collection"
        if (self.currDirectory is None) or (self.numPictures == -1):
            print("ERROR: cannot run makeCollection without a directory and a number of pictures")
            return
        self.logger.log("Reading in image database")


        for i in range(self.numPictures):
            picNum = self.startPicture + i
            image = self.getFileByNumber(picNum)
            if self.height == 0:
                self.height, self.width, depth = image.shape
            #self.logger.log("Image = " + str(picNum))
            features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
            self.featureCollection[picNum] = features
            if i % 100 == 0:
                print i

        self.logger.log("Length of collection = " + str(self.numPictures))


    # ------------------------------------------------------------------------
    # One of the major operations we can undertake, comparing all pairs in a range
    def matchImage(self, camImage):
        if len(self.featureCollection) == 0:
            print("ERROR: must have built collection before running this.")
            return
        self.logger.log("Choosing frames from video to compare to collection")

        features = ImageFeatures.ImageFeatures(camImage, 9999, self.logger, self.ORBFinder)
        cv2.imshow("Primary image", camImage)
        cv2.moveWindow("Primary image", 0, 0)
        features.displayFeaturePics("Primary image features", 0, 0)
        self._findBestNMatches(features, self.numMatches)


    def _findBestNMatches(self, features, numMatches):
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
        for j in range(len(bestZipped)):
            (nextScore, nextMatch) = bestZipped[j]
            cv2.imshow("Match Picture", nextMatch.getImage())
            cv2.moveWindow("Match Picture", self.width+10, 0)
            nextMatch.displayFeaturePics("Match Picture Features", self.width+10, 0)
            self.logger.log("Image " + str(nextMatch.getIdNum()) + " matches with similarity = " + str(nextScore))
            cv2.waitKey(0)



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
    rospy.init_node('ImageMatching')
    print "GOT HERE"
    matcher = ImageMatcher(logFile = True, logShell = True,
                           dir1 = "/home/macalester/Desktop/githubRepositories/catkin_ws/src/match_seeker/res/Feb2017Data/",
                           baseName = "frame",
                           ext = "jpg",
                           startPic = 0,
                           numPics = 500)
    print "finish ImageMatcher call"
    matcher.run()
    rospy.on_shutdown(matcher.exit)
    rospy.spin()



