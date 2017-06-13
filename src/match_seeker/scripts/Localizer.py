
import math
import cv2
# from espeak import espeak

from DataPaths import basePath, imageDirectory, locData
import ImageDataset


class Localizer(object):


    def __init__(self, bot, mapGraph, logger):
        self.robot = bot
        self.olin = mapGraph
        self.logger = logger

        self.lostCount = 0
        self.beenGuessing = False

        self.lastKnownLoc = None
        self.confidence = 0

        self.dataset = ImageDataset.ImageDataset(logger, numMatches = 3)
        self.dataset.setupData(basePath + imageDirectory, basePath + locData, "frame", "jpg")



    def findLocation(self, cameraIm):
        """Given the current camera image, update the robot's estimated current location, the confidence associated
        with that location. This method returns None if the robot is not at a "significant" location (one of the points
         in the olin graph) with a high confidence. If the robot IS somewhere significant and confidence is high
         enough, then the information about the location is returned so the planner can respond to it."""

        matches = self.dataset.matchImage(cameraIm, self.lastKnownLoc, self.confidence)

        self._displayMatch(matches[0])
        if matches[0][0] > 90:
            self.logger.log("I have no idea where I am.")
            self.lostCount += 1
            self.beenGuessing = False
            if self.lostCount >= 5:
                self.lastKnownLoc = None
                self.confidence = 0
            return None
        else:
            self.lostCount = 0


            guess, head, probLoc, conf = self._guessLocation(matches)
            self.logger.log("I think I am at node " + str(guess) + ", and I am " + conf)

            if conf == "very confident." or conf == "close, but guessing.":
                # print "I found my location."
                return guess, head, probLoc, "at node"
            elif conf == "confident, but far away.":
                return guess, head, probLoc, "check coord"
            else:
                return None


    def _displayMatch(self, match):
        """Given match information, of the form (score, ImageFeatures), it displays the match with the score
        written in the lower left corner."""
        (score, imFeat) = match
        matchIm = imFeat.getImage()
        matchIm = matchIm.copy()
        cv2.putText(matchIm, str(score), (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0))
        cv2.imshow("Match Picture", matchIm)
        # cv2.moveWindow("Match Picture", self.width + 10, 0)
        cv2.waitKey(20)



    def _guessLocation(self, bestMatches):
        (lowScore, lowFeat) = bestMatches[0]
        idNum = lowFeat.getIdNum()
        bestX, bestY, bestHead = self.dataset.getLoc(idNum)
        (bestNodeNum, nodeX, nodeY, bestDist) = self._findClosestNode((bestX, bestY))

        self.logger.log("This match is tagged at " + str(bestX) + ", " + str(bestY) + ".")
        self.logger.log("The closest node is " + str(bestNodeNum) + " at " + str(bestDist) + "  meters.")

        if bestMatches[0][0] < 70:
            self.beenGuessing = False
            if bestDist <= 0.8:
                # espeak.synth(str(nodeNum))
                self.lastKnownLoc = (bestX, bestY)
                self.confidence = 10.0
                return bestNodeNum, bestHead, (bestX,bestY), "very confident."
            else:
                self.lastKnownLoc = (bestX, bestY)
                if self.beenGuessing:
                    self.confidence = max(0.0, self.confidence - 0.5)
                else:
                    self.confidence = 10.0
                    self.beenGuessing = True
                return bestNodeNum, bestHead, (bestX,bestY), "confident, but far away."
        else:
            guessNodes = [bestNodeNum]
            dist = 0
            for j in range(1, len(bestMatches)):
                (nextScore, nextMatch) = bestMatches[j]
                idNum = nextMatch.getIdNum()
                locX, locY, locHead = self.dataset.getLoc(idNum)
                (nodeNum, x, y, dist) = self._findClosestNode((locX, locY))
                if nodeNum not in guessNodes:
                    guessNodes.append(nodeNum)
            if len(guessNodes) == 1 and bestDist <= 0.8:
                self.lastKnownLoc = (bestX, bestY)
                if self.beenGuessing:
                    self.confidence = max(0.0, self.confidence - 0.5)
                else:
                    self.confidence = 10.0
                    self.beenGuessing = True
                return guessNodes[0], bestHead, (bestX,bestY), "close, but guessing."
            elif len(guessNodes) == 1:
                self.lastKnownLoc = (bestX, bestY)
                self.confidence = 5.0
                self.beenGuessing = False
                return guessNodes[0], bestHead, (bestX,bestY), "far and guessing."
            else:
                nodes = str(guessNodes[0])
                for i in range(1,len(guessNodes)):
                    nodes += " or " + str(guessNodes[i])
                self.confidence = 0.0
                self.beenGuessing = False
                # TODO: figure out if bestHead is the right result, but for now it's probably ignored anyway
                return nodes, bestHead, (bestX,bestY), "totally unsure."



    def _findClosestNode(self, (x, y)):
        """uses the location of a matched image and the distance formula to determine the node on the olingraph
        closest to each match/guess"""
        closestNode = None
        closestX = None
        closestY = None
        bestVal = None
        for nodeNum in self.olin.getVertices():
            if closestNode is None:
                closestNode = nodeNum
                closestX, closestY = self.olin.getData(nodeNum)
                bestVal = self._euclidDist( (closestX, closestY), (x, y) )
            (nodeX,nodeY) = self.olin.getData(nodeNum)
            val = self._euclidDist( (nodeX, nodeY), (x, y) )
            if (val <= bestVal):
                bestVal = val
                closestNode = nodeNum
                closestX, closestY = (nodeX,nodeY)
        return (closestNode, closestX, closestY, bestVal)



    def _euclidDist(self, (x1, y1), (x2, y2)):
        """Given two tuples containing two (x, y) points, this computes the straight-line distsance
        between the two points"""
        return math.sqrt( (x2 - x1) ** 2 + (y2 - y1) ** 2 )
