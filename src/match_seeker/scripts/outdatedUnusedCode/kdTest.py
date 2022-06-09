
import os
import cv2
import OutputLogger
import numpy as np
from scipy import spatial
import ImageFeatures
from DataPaths import basePath,imageDirectory,locData
import MapGraph


class ImageMatcher(object):
    """..."""


    def __init__(self, logFile = False, logShell = False,
                 dir1 = None,locFile = None, baseName = 'foo', ext= "jpg",
                 numMatches = 4):
        self.logToFile = logFile
        self.logToShell = logShell
        self.currDirectory = dir1
        self.locFile = locFile
        self.baseName = baseName
        self.currExtension = ext
        self.numMatches = numMatches
        self.threshold = 800.0
        self.height = 0
        self.width = 0
        self.logger = OutputLogger.OutputLogger(self.logToFile, self.logToShell)

        # self.robot = bot

        # Add line to debug ORB
        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()

        self.featureCollection = {} # dict with key being image number and value being ImageFeatures
        self.numByLoc = {}
        self.locByNum = {}
        self.tree = None
        self.xyArray = []

        self.path = basePath + "scripts/olinGraph.txt"
        self.olin = MapGraph.readMapFile(self.path)

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
            if file[-3:] != 'jpg':
                print "FOUND NON IMAGE IN IMAGE FOLDER:", file
                continue
            image = cv2.imread(self.currDirectory + file)
            end = len(file) - (len(self.currExtension) + 1)
            picNum = int(file[len(self.baseName):end])
            if self.height == 0:
                self.height, self.width, depth = image.shape
            features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
            if picNum in self.featureCollection:
                print "ERROR: duplicate number", picNum
            self.featureCollection[picNum] = features
        self.logger.log("Length of collection = " + str(len(self.featureCollection)))

    def makeLocDict(self):
        # picNum to x,y,heading
        file = open(self.locFile)
        for line in file.readlines():
            line = line.rstrip('/n')
            line = line.split()
            self.locByNum[int(line[0])] = [float(x) for x in line[1:]]

        # x,y to picNums w/ headings
        for picNum in self.locByNum:
            (x,y,h) = self.locByNum[picNum]
            if (x,y) in self.numByLoc:
                self.numByLoc[(x,y)].append(picNum)
            else:
                self.numByLoc[(x,y)] = [picNum]

    def buildKDTree(self):
        xyKeys = self.numByLoc.keys()
        self.xyArray = np.asarray(xyKeys)
        self.tree = spatial.KDTree(self.xyArray)
        # print self.tree.data




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
        return self._findBestNMatches(features, self.numMatches)


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
        print [x[0]for x in bestZipped]
        self.logger.log("==========Location Update==========")



        if bestZipped[0][0] > 90:
            self.logger.log("I have no idea where I am.")
        else:
            print "may know where I am"
            im =  bestZipped[0][1].getImage()
            locX,locY,locHead = self.locByNum[bestZipped[0][1].getIdNum()]
            (num, x, y, distSq) = self.findClosestNode((locX, locY))
            self.logger.log("This match is tagged at " + str(locX) + ", " + str(locY) + ".")
            self.logger.log("The closest node is " + str(num) + " at " + str(distSq) + " sq meters.")
            cv2.imshow("Match Picture", im)
            cv2.moveWindow("Match Picture", self.width + 10, 0)
            cv2.waitKey(20)
            # img = self.img.copy()
            # (self.mapHgt, self.mapWid, dep) = img.shape
            # cv2.imshow("map",img)

            guess, pts, head, conf = self.guessLocation(bestZipped)
            self.logger.log("I think I am at node " + str(guess) + ", and I am " + conf)

            if conf == "very confident." or conf == "close, but guessing.":
                print "I found my location."
                return guess, pts, head
            else:
                return None


    def guessLocation(self,bestZipped):
        worst = len(bestZipped)-1
        if bestZipped[0][0] < 70:
            match = bestZipped[0][1]
            idNum = match.getIdNum()
            bestX, bestY, bestHead = self.locByNum[idNum]
            (nodeNum, x, y, distSq) = self.findClosestNode((bestX, bestY))
            if distSq <= 0.8:
                return nodeNum, (x,y), bestHead, "very confident."
            else:
                return nodeNum, (x,y), bestHead, "confident, but far away."
        else:
            guessNodes = []
            distSq = 0
            for j in range(len(bestZipped) - 1, -1, -1):
                (nextScore, nextMatch) = bestZipped[j]
                idNum = nextMatch.getIdNum()
                locX, locY, locHead = self.locByNum[idNum]
                (nodeNum, x, y, distSq) = self.findClosestNode((locX, locY))
                if nodeNum not in guessNodes:
                    guessNodes.append(nodeNum)
            if len(guessNodes) == 1 and distSq <= 0.8:
                return guessNodes[0], (x,y), locHead, "close, but guessing."
            elif len(guessNodes) == 1:
                return guessNodes[0], (x,y), locHead, "far and guessing."
            else:
                nodes = str(guessNodes[0])
                for i in range(1,len(guessNodes)):
                    nodes += " or " + str(guessNodes[i])
                return nodes, (x,y), locHead, "totally unsure."


    def findClosestNode(self, (x, y)):
        """uses the location of a matched image and the distance formula to determine the node on the olingraph
        closest to each match/guess"""
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
        return (closestNode, closestX, closestY, bestVal)


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





if __name__ == '__main__':
    # REMEMBER THIS IS TEST CODE ONLY!
    # change paths for files in OSPathDefine
    matcher = ImageMatcher(logFile = True, logShell = True,
                           dir1 =basePath + imageDirectory,
                           locFile = basePath + locData,
                           baseName = "frames",
                           ext = "jpg", numMatches=3)
    matcher.makeCollection()
    matcher.makeLocDict()
    matcher.buildKDTree()

    while (True):
        userInput = raw_input("x,y ")
        if userInput == "q":
            break
        rad = raw_input("radius: ")
        loc=userInput.split(",")
        x = float(loc[0])
        y = float(loc[1])
        rad = float(rad)

        pt = np.array([[x,y]])
        i = matcher.tree.query_ball_point(pt,rad)

        # print "d", d
        print "i", i

        nums=[]

        for loc in matcher.xyArray[i[0]]:
            tup = tuple(loc)
            nums.extend(matcher.numByLoc[tup])
            # print loc
        print nums

        for num in nums:
            feat = matcher.featureCollection[num]
            cv2.imshow("feat", feat.getImage())
            cv2.waitKey(0)
        # print "distance", d[0]
        # print "coord", matcher.xyArray[i[0]]

















