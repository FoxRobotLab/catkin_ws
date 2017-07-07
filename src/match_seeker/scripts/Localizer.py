
import math
import cv2
# from espeak import espeak

from DataPaths import basePath, imageDirectory, locData
import ImageDataset
import MonteCarloLocalize


class Localizer(object):


    def __init__(self, bot, mapGraph, logger):
        self.robot = bot
        self.olin = mapGraph
        self.logger = logger

        self.lostCount = 0
        self.beenGuessing = False

        self.lastKnownLoc = None
        self.confidence = 0
        self.currLoc = None

        self.dataset = ImageDataset.ImageDataset(logger, numMatches = 3)
        self.dataset.setupData(basePath + imageDirectory, basePath + locData, "frame", "jpg")

        self.mcl = MonteCarloLocalize.monteCarloLoc()
        self.mcl.initializeParticles(500)
        self.mcl.getOlinMap(basePath + "res/map/olinNewMap.txt")

        self.odomScore = 100.0




    def findLocation(self, cameraIm):
        """Given the current camera image, update the robot's estimated current location, the confidence associated
        with that location. This method returns None if the robot is not at a "significant" location (one of the points
         in the olin graph) with a high confidence. If the robot IS somewhere significant and confidence is high
         enough, then the information about the location is returned so the planner can respond to it."""

        odomInfo = self.odometer()
        odomLoc = (odomInfo[2][0], odomInfo[2][1], odomInfo[1])

        # if odomInfo[3] <= 1.0:
        #     return "at node", (odomInfo[0], odomInfo[1], odomInfo[2])
        # else:
        #     return "check coord", (odomInfo[0], odomInfo[1], odomInfo[2])

        if self.lastKnownLoc is None:
            lklSt = "Last known loc: None so far   confidence = {0:4.2f}"
            self.logger.log( lklSt.format(self.confidence) )
        else:
            lklSt = "Last known loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})   confidence = {3:4.2f}"
            (x, y, h) = self.lastKnownLoc
            self.logger.log( lklSt.format(x, y, h, self.confidence) )

        matches = self.dataset.matchImage(cameraIm, self.lastKnownLoc, self.confidence)
        (bestScore, bestFeat) = matches[0]
        bestPicNum = bestFeat.getIdNum()
        bestX, bestY, bestHead = self.dataset.getLoc(bestPicNum)
        self.currLoc = (bestX, bestY)

        self._displayMatch(bestPicNum, bestScore, bestFeat)
        picLocSt = "Best image loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})   score = {3:4.2f}"
        self.logger.log( picLocSt.format(bestX, bestY, bestHead, bestScore) )

        matchLocs = []
        matchScores = []
        for i in range(len(matches)):
            (nextScore, nextMatch) = matches[i]
            idNum = nextMatch.getIdNum()
            locX, locY, locHead = self.dataset.getLoc(idNum)
            matchLocs.append((locX, locY,locHead))
            matchScores.append(nextScore)

        moveInfo = self.robot.getTravelDist()

        mclData = {'matchPoses': matchLocs,
                   'matchScores': matchScores,
                   'odomPose': odomLoc,
                   'odomScore': self.odomScore}
        comPose = self.mcl.mclCycle(mclData, moveInfo)
        (centerX, centerY, centerHead) = comPose
        centerStr = "CENTER OF PARTICLE MASS: ({0: 4.2f}, {1:4.2f}, {2:4.2f})"
        self.logger.log(centerStr.format(centerX, centerY, cenerHead))



        if bestScore < 5: #TODO:changed from >90
            self.logger.log("      I have no idea where I am.     Lost Count = " + str(self.lostCount))
            self.lostCount += 1
            self.beenGuessing = False
            if self.lostCount == 5:
                self.lastKnownLoc = None
                self.confidence = 0
            elif self.lostCount >= 10:
                return "look", None
            return "continue", None
        else:
            self.lostCount = 0

            guess, conf = self._guessLocation(bestScore, self.currLoc, bestHead, matches)
            matchInfo = (guess, self.currLoc, bestHead)
            self.logger.log("      Nearest node: " + str(guess) + "  Confidence = " + str(conf))

            if conf == "very confident." or conf == "close, but guessing.":
                return "at node", matchInfo
            elif conf == "confident, but far away.":
                return "check coord", matchInfo
            else:
                # Guessing but not close...
                return "keep-going", matchInfo


    def _displayMatch(self, bestPicNum, bestScore, bestFeat):
        """Given match information, of the form (score, ImageFeatures), it displays the match with the score
        written in the lower left corner."""
        dispTemplate = "{0:d}: {1:3.1f}"
        dispString = dispTemplate.format(bestPicNum, bestScore)
        matchIm = bestFeat.getImage()
        matchIm = matchIm.copy()
        cv2.putText(matchIm, dispString, (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0))
        cv2.imshow("Match Picture", matchIm)
        # cv2.moveWindow("Match Picture", self.width + 10, 0)
        cv2.waitKey(20)



    def _guessLocation(self, bestScore, bestLoc, bestHead, bestMatches):
        (bestX, bestY) = bestLoc
        (bestNodeNum, nodeX, nodeY, bestDist) = self._findClosestNode(bestLoc)
        closeDistStr = "      The closest node is {0:d} at {1:4.2f} meters."
        self.logger.log( closeDistStr.format(bestNodeNum, bestDist) )

        odoUpdateStr = "UPDATING ODOMETRY TO: ({0:4.2f}, {1:4.2f}, {2:4.2f})"
        if bestScore > 30:
            self.beenGuessing = False
            if bestDist <= 0.8:
                self.lastKnownLoc = (bestX, bestY, bestHead)
                self.confidence = 10.0
                if self.odomScore < 50 and bestScore >= self.odomScore:
                    self.logger.log( odoUpdateStr.format(bestX, bestY, bestHead) )
                    self.odomScore = 100.0
                    self.robot.updateOdomLocation(bestX, bestY, bestHead)
                return bestNodeNum, "very confident."
            else:
                self.lastKnownLoc = (bestX, bestY, bestHead)
                if self.odomScore < 50 and bestScore > 90:
                    self.logger.log( odoUpdateStr.format(bestX, bestY, bestHead) )
                    self.odomScore = 100.0
                    self.robot.updateOdomLocation(bestX, bestY, bestHead)
                if self.beenGuessing:
                    self.confidence = max(0.0, self.confidence - 0.5)
                else:
                    self.confidence = 10.0
                    self.beenGuessing = True
                return bestNodeNum, "confident, but far away."
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
                self.lastKnownLoc = (bestX, bestY, bestHead)
                if self.beenGuessing:
                    self.confidence = max(0.0, self.confidence - 0.5)
                else:
                    self.confidence = 10.0
                    self.beenGuessing = True
                return guessNodes[0], "close, but guessing."
            elif len(guessNodes) == 1:
                self.lastKnownLoc = (bestX, bestY, bestHead)
                self.confidence = 5.0
                self.beenGuessing = False
                return guessNodes[0], "far and guessing."
            else:
                nodes = str(guessNodes[0])
                for i in range(1,len(guessNodes)):
                    nodes += " or " + str(guessNodes[i])
                self.confidence = 0.0
                self.beenGuessing = False
                # TODO: figure out if bestHead is the right result, but for now it's probably ignored anyway
                return nodes, "totally unsure."


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


    def odometer(self):
        formStr = "Odometer loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})  confidence = {3:4.2f}"
        x, y, yaw = self.robot.getOdomData()
        (nearNode, closeX, closeY, bestVal) = self._findClosestNode((x,y))
        self.logger.log( formStr.format(x, y, yaw, self.odomScore) )
        self.odomScore = max(0.001, self.odomScore - 0.5)

        if self.robot.hasWheelDrop():
            self.odomScore = 0.001

        return nearNode, yaw, (x, y), bestVal


    def setLastLoc(self, prevDestXY):
        (x, y) = prevDestXY
        if self.lastKnownLoc == None:
            self.lastKnownLoc = (x, y, 0)
        else:
            (oldX, oldY, oldH) = self.lastKnownLoc
            self.lastKnownLoc = (x, y, oldH)


    def _euclidDist(self, (x1, y1), (x2, y2)):
        """Given two tuples containing two (x, y) points, this computes the straight-line distsance
        between the two points"""
        return math.sqrt( (x2 - x1) ** 2 + (y2 - y1) ** 2 )
