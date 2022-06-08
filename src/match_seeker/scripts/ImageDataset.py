#!/usr/bin/env python

""" ========================================================================
ImageDataset.py

Created: June 2017
Author: Susan

This creates the ImageDataset object, which reads in a set of images and a
location file, and creates dictionaries and a KDTree to hold the images
and organize them by their locations. It provides methods for getting some number
of closest matches.
======================================================================== """
import os

import sys

import numpy as np
from scipy import spatial
import cv2

import ImageFeatures
#sys.path.insert(0, "/home/macalester/PycharmProjects/src/deep_learning/olri_classifier/olin_factory")
#from olin_factory import paths, cell
from src.match_seeker.scripts.olri_classifier.DataManipulations import olin_factory


class ImageDataset(object):
    """This constructs a dataset to associate images with locations, allowing us to access the location of an
    image, and also to select images that lie within some boundary of a given location."""


    def __init__(self, logWriter, gui, numMatches = 4):
        self.numMatches = numMatches
        self.threshold = 800.0

        self.logger = logWriter
        self.gui = gui

        # Add line to debug ORB
        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()

        self.featureCollection = {} # dict with key being image number and value being ImageFeatures
        self.numByLoc = {}
        self.locByNum = {}
        self.tree = None
        self.xyArray = None

        # TODO: redundant w/ OlinWorldMap._readGraphMap --> merge
        cellCenterF = open(olin_factory.paths.cell_graph_path, "r")
        cellCenterDict = dict()
        for line in cellCenterF:
            if line.startswith("Markers"):
                break
            if line[0] == '#' or line.isspace() or line.startswith("N") or (not line[0] in "0123456789"):
                continue


            parts = line.split()
            cellNum = parts[0]
            parts[1]=parts[1].lstrip("(")
            parts[1]=parts[1].rstrip(",")
            parts[2]=parts[2].rstrip(")")
            centerX, centerY = [float(v) for v in parts[1:]]
            cellCenterDict[cellNum] = [centerX, centerY]
        self.cellCenterDict = cellCenterDict




    def setupData(self, currDir, locFile, baseName, ext):
        """This method reads in the data and makes the various data structures."""
        self.makeCollection(currDir, baseName, ext)
        self.makeLocDict(locFile)
        self.buildKDTree()


    def makeCollection(self, currDir, baseName, ext):
        """Reads in all the images in the specified directory, start number and end number, and
        makes a list of ImageFeature objects for each image read in."""
        if (currDir is None):
            self.logger.log("ERROR: cannot run makeCollection without a directory")
            return
        self.logger.log("Reading image dataset...")
        self.gui.updateMessageText("Reading image dataset...")
        listDir = os.listdir(currDir)
        cnt = 0
        for filename in listDir:
            # Check if the next file is the right form, and extract the number part, if it is
            (picNum, foundNum) = self._extractNumber(filename, baseName, ext)
            if not foundNum:
                continue
            image = cv2.imread(currDir + filename)
            features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
            if picNum in self.featureCollection:
                self.logger.log("ERROR: duplicate number: " +  str(picNum))
            self.featureCollection[picNum] = features
            if cnt % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            cnt += 1
        # self.logger.log("Length of collection = " + str(len(self.featureCollection)))


    def makeLocDict(self, locFile):
        """Creates a dictionary mapping picture numbers to their locations (a list of [x, y, heading]) and
        also creates a dictionary mapping (x, y) locations to the picture numbers at those locations."""
        # picNum to x,y,heading
        filename = open(locFile, 'r')
        for line in filename:
            # line = line.strip()
            line = line.split()
            loc = int(line[0])
            x = float(line[1])
            y = float(line[2])
            h = float(line[3])
            self.locByNum[loc] = [x, y, h]
        filename.close()

        # x,y to picNums w/ headings
        for picNum in self.locByNum:
            (x, y, h) = self.locByNum[picNum]
            if (x, y) in self.numByLoc:
                self.numByLoc[(x, y)].append(picNum)
            else:
                self.numByLoc[(x, y)] = [picNum]


    def buildKDTree(self):
        """Assuming that the two dictionaries mapping picture nums to locations and vice versa are already
        built, this constructs a KDTree for the (x, y) locations of the dataset."""
        xyKeys = self.numByLoc.keys()
        # print("ImageDataset.buildKDTree: xyKeys", xyKeys)
        self.xyArray = np.asarray(xyKeys)
        leafSize=10
        self.tree = spatial.KDTree(self.xyArray, leafsize=leafSize)
        self.logger.log("<<<ImageDataset.buildKDTree>>> KDTree built with leafsize="+str(leafSize))
        # print("ImageDataset.buildKDTree: tree", self.tree)



    def _extractNumber(self, filename, baseName, ext):
        """Pulls apart the filename to make sure it is the right format. If it is, then the picture number is returned.
        If it is no, then False is returned."""
        baseLen = len(baseName)
        extLen = len(ext)
        extStart = len(filename) - (extLen + 1)
        basePart = filename[:baseLen]
        numPart = filename[baseLen:extStart]
        extPart = filename[extStart:]
        if (extPart != '.' + ext) or (basePart != baseName) or (not numPart.isdigit()):
            self.logger.log("<<<ImageDataset._extractNumber>>> ERROR: Found wrong format file in image folder: " + filename)
            return None, False
        else:
            return int(numPart), True


    def getLoc(self, picNum):
        """Given a picture number, return the location that is associated with it."""
        return self.locByNum[picNum]


    # ------------------------------------------------------------------------
    # matches a new image to the dataset
    def matchImage(self, camImage, lastKnown, confidence):
        """Given an image from the robot's camera, the last known location of the robot, and
        how confident the robot is about that location, this examines a subset of its dataset of images and finds
        the best matches among that subset. """
        if len(self.featureCollection) == 0:
            self.logger.log("<<<ImageDataset.matchImage>>> ERROR: must have built collection before running this.")
            return
        features = ImageFeatures.ImageFeatures(camImage, 9999, self.logger, self.ORBFinder)
        # print("ImageDataset.matchImage: features: ", features) #DEBUG

        bestScores, bestMatches = self._findBestNMatches(features, lastKnown, confidence)
        # print("ImageDataset.matchImage: bestScores, bestMatches: ", bestScores, bestMatches) #DEBUG
        matchLocs = []
        for i, match in enumerate(bestMatches):
            idNum = match.getIdNum()
            loc = self.getLoc(idNum)
            matchLocs.append(loc)
            self.logger.log("<<<ImageDataset.matchImage>>> matchLoc"+str(i)+"="+str(loc))  # DEBUG
        bestX, bestY, bestHead = matchLocs[0]

        # self._displayMatch(features, bestScores[0], bestMatches[0])
        features.evaluateSimilarity(bestMatches[0],verbose=True)
        picLocSt = "best match loc=({0:4.2f}, {1:4.2f}, {2:4.2f}), score={3:4.2f}"
        self.logger.log("<<<ImageDataset.matchImage>>> "+picLocSt.format(bestX, bestY, bestHead, bestScores[0]))
        return bestScores, matchLocs


    def _findBestNMatches(self, currImFeatures, lastKnown, confidence):
        """Looks through the collection of features and keeps the numMatches
        best matches. It zips them together"""
        potentialMatches = self._findPotentialMatches(lastKnown, confidence)
        #TODO: Changed to use cnn
        #potentialMatches = self._findCNNMatches(cellNum=lastKnown, radius=30.0)
        # print("ImageDataset._findBestNMatches(): potentialMatches", potentialMatches)
        (bestScores, bestMatches) = self._selectBestN(potentialMatches, currImFeatures)
        # print("ImageDataset._findBestNMatches(): (bestScores, bestMatches)", bestScores, bestMatches)

        formSt = "{0:4.2f}"
        self.logger.log("<<<ImageDataset._findBestNMatches>>> Best matches have scores: " + " ".join([ formSt.format(x) for x in bestScores]))
        return  bestScores, bestMatches


    def _findPotentialMatches(self, lastKnown, confidence):
        """Computes the set of possible matches to be examined. In the worst case, this is the entire image
        dataset. However, it tries to reduce this using the KDTree to get only pictures within some range."""
        radius = self._confidenceToRadius(confidence)
        potentialMatches = []
        if lastKnown is None or confidence == 0.0:
            potentialMatches = self.featureCollection.keys()
        else:
            pt = np.array(lastKnown[:2])
            potentialMatches = self.getNearPos(pt, radius)
            if potentialMatches == []:
                potentialMatches = self.featureCollection.keys()
        # self.logger.log("Potential matches length: " + str(len(potentialMatches)))
        # self.logger.log("      Searching with radius " + str(radius))
        self.logger.log("<<<ImageDataset._findPotentialMatches>>> search radius=" + str(radius))
        self.logger.log("<<<ImageDataset._findPotentialMatches>>> lastKnown=" + str(lastKnown))
        # self.logger.log("<<<ImageDataset._findPotentialMatches>>> potential matches=" + str(potentialMatches))

        self.gui.updateRadius(radius)
        return potentialMatches


    def _findCNNMatches(self, cellNum, radius):
        centerX, centerY = self.cellCenterDict[str(cellNum)]
        potentialMatches = self.getNearPos(pos=(centerX, centerY), radius=radius)
        # self.logger.log("      Searching with radius (cnn) " + str(radius))
        self.logger.log("<<<ImageDataset._findCNNMatches>>> search radius=" + str(radius))
        self.logger.log("<<<ImageDataset._findCNNMatches>>> lastKnown=" + str(cellNum))
        # self.logger.log("<<<ImageDataset._findCNNMatches>>> potential matches=" + str(potentialMatches))
        # print("In ImageDataset:_findCNNMAtches: searching with (cnn) radius " +str(radius))
        # print("In ImageDataset:_findCNNMAtches: potentialMatches ", potentialMatches)

        self.gui.updateRadius(radius)
        return potentialMatches


    def _selectBestN(self, potentialMatches, currFeat):
        """Takes in a list of potential matching images (image numbers, specifically) from the dataset, and
        it checks their similarity to the current image. Keeps the best numMatches """
        bestMatches = []
        bestScores = []

        for pos in potentialMatches:
            feat = self.featureCollection[pos]
            simValue = currFeat.evaluateSimilarity(feat)
            if simValue < self.threshold:
                self._doubleInsert(simValue, feat, bestScores, bestMatches)
                if len(bestScores) > self.numMatches:
                    bestScores.pop()
                    bestMatches.pop()
        return bestScores, bestMatches


    def _doubleInsert(self, score, feat, scoreList, featList):
        """Given a score and a feature, inserts it to keep the scorelist in increasing order, inserting the feature in the
        feature list at the same point."""
        indx = 0
        while indx < len(scoreList) and (score < scoreList[indx]): #TODO:changed the second part from >
            indx += 1

        if indx == len(scoreList):
            scoreList.append(score)
            featList.append(feat)
        else:
            scoreList.insert(indx, score)
            featList.insert(indx, feat)


    # def _displayMatch(self, camImFeat, bestScore, bestMatchFeat):
    #     """Given match information, of the form (score, ImageFeatures), it displays the match with the score
    #     written in the lower left corner."""
    #     dispTemplate = "{0:3.1f}"
    #     dispString = dispTemplate.format(bestScore)
    #     # matchIm = bestMatchFeat.getImage()
    #     # matchIm = matchIm.copy()
    #     ImageFeatures.ImageFeatures.displayFeaturePics(bestMatchFeat, camImFeat, "Match Picture")
    #     cv2.putText(matchIm, dispString, (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0))
    #     cv2.imshow("Match Picture", matchIm)
    #     # cv2.moveWindow("Match Picture", self.width + 10, 0)
    #     cv2.waitKey(20)

    def _confidenceToRadius(self, confidence):
        """Maps the confidence level that comes from outside into a radius for use in searching for
        potential matches.s
        Confidence:   1  2  3  4  5  6  7  8  9  10
        Radius:      15                           3 """
        perc = 1.0 - (confidence / 10.0)
        # radius = (perc * 12.0) + 3.0
        radius = (perc * 4.0) + 2.0
        # if radius >= 6:
        #     radius = 6
        return radius

    def getNearPos(self, pos, radius):
        """Given an (x, y) position and a radius, return all picture numbers that are within radius of
         that position."""

        #print(pos)
        res = self.tree.query_ball_point(pos, radius) #TODO: problem here
        #print("<<<ImageDataset.getNearPos>>> tree query with pos="+str(pos)+ " radius="+str(radius) + str(res))
        xyCoords = [self.xyArray[indx] for indx in res]
        # print("ImageDataset.getNearPos(): xyCoords: ", xyCoords)
        imageList = []
        for loc in xyCoords:
            #print("ImageDataset.getNearPos(): xyCoords loc: ", loc)
            tupLoc = tuple(loc)
            locsAtPos = self.numByLoc[tupLoc]
            imageList.extend(locsAtPos)
        return imageList


#
#
# if __name__ == '__main__':
#     # Demo code
#     matcher = ImageDataset(None, None, 3)
#     matcher.setupData(basePath + ImageDirectory, locData, "frame", "jpg")

