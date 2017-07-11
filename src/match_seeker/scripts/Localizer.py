
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

        self.lastKnownLoc = None
        self.confidence = 0

        self.dataset = ImageDataset.ImageDataset(logger, numMatches = 3)
        self.dataset.setupData(basePath + imageDirectory, basePath + locData, "frame", "jpg")

        self.mcl = MonteCarloLocalize.monteCarloLoc(self.olin)
        self.mcl.initializeParticles(500)
        # self.mcl.getOlinMap(basePath + "res/map/olinNewMap.txt")

        self.odomScore = 100.0



    def findLocation(self, cameraIm):
        """Given the current camera image, update the robot's estimated current location, the confidence associated
        with that location. This method returns None if the robot is not at a "significant" location (one of the points
         in the olin graph) with a high confidence. If the robot IS somewhere significant and confidence is high
         enough, then the information about the location is returned so the planner can respond to it."""
        if self.lastKnownLoc is None:
            lklSt = "Last known loc: None so far   confidence = {0:4.2f}"
            self.logger.log( lklSt.format(self.confidence) )
        else:
            lklSt = "Last known loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})   confidence = {3:4.2f}"
            (x, y, h) = self.lastKnownLoc
            self.logger.log( lklSt.format(x, y, h, self.confidence) )

        odomLoc = self.odometer()

        moveInfo = self.robot.getTravelDist()
        scores, matchLocs = self.dataset.matchImage(cameraIm, self.lastKnownLoc, self.confidence)

        mclData = {'matchPoses': matchLocs,
                   'matchScores': scores,
                   'odomPose': odomLoc,
                   'odomScore': self.odomScore}


        comPose, var = self.mcl.mclCycle(mclData, moveInfo)
        (centerX, centerY, centerHead) = comPose
        centerStr = "CENTER OF PARTICLE MASS: ({0: 4.2f}, {1:4.2f}, {2:4.2f}), VARIANCE: ({3:4.2f})"
        self.logger.log(centerStr.format(centerX, centerY, centerHead, var))

        bestScore = scores[0]
        bestX, bestY, bestHead = matchLocs[0]

        odoUpdateStr = "UPDATING ODOMETRY TO: ({0:4.2f}, {1:4.2f}, {2:4.2f})"
        if self.odomScore < 50 and var < 5.0:
            self.logger.log(odoUpdateStr.format(centerX,centerY, centerHead))
            self.odomScore = 60
            self.robot.updateOdomLocation(centerX, centerY, centerHead)
        elif self.odomScore < 50 and bestScore >= 50:
            self.logger.log(odoUpdateStr.format(bestX, bestY, bestHead))
            self.odomScore = bestScore
            self.robot.updateOdomLocation(bestX, bestY, bestHead)


        if var < 10.0:
           return self.mclResponse(comPose, var)
        else:
            return self.matchResponse(matchLocs, scores)


    def mclResponse(self, comPose, var):
        (bestNodeNum, nodeX, nodeY, bestDist) = self.olin.findClosestNode(comPose)
        mclInfo = (bestNodeNum, comPose)
        self.lastKnownLoc = comPose
        if var < 5.0:
            self.confidence = 10.0
        elif var < 10.0:
            self.confidence = 5.0
        else:
            self.confidence = max(0.0, self.confidence-0.5)

        if bestDist <= 0.8:
            return "at node", mclInfo
        else:
            return "check coord", mclInfo


    def matchResponse(self,matchLocs, scores):
        """a messy if statement chain that takes in match scores and locations and spits out arbitrary information that
        manages behavior in matchPlanner"""

        bestScore = scores[0]
        bestLoc = matchLocs[0]

        if bestScore < 5: #changed from >90
            self.logger.log("      I have no idea where I am.     Lost Count = " + str(self.lostCount))
            self.lostCount += 1
            if self.lostCount == 5:
                self.setLocation("lost", None)
            elif self.lostCount >= 10:
                return "look", None
            return "continue", None
        else:
            self.lostCount = 0
            guess, conf = self._guessLocation(bestScore, bestLoc, matchLocs)
            matchInfo = (guess, bestLoc)
            self.setLocation(conf, bestLoc)
            self.logger.log("      Nearest node: " + str(guess) + "  Confidence = " + str(conf))

            if conf == "very confident." or conf == "close, but guessing.":
                self.mcl.scatter(matchInfo[1])
                return "at node", matchInfo
            elif conf == "confident, but far away.":
                return "check coord", matchInfo
            else:
                # Guessing but not close...
                return "keep-going", matchInfo


    def _guessLocation(self, bestScore, bestLoc, matchLocs):
        (bestNodeNum, nodeX, nodeY, bestDist) = self.olin.findClosestNode(bestLoc)
        closeDistStr = "      The closest node is {0:d} at {1:4.2f} meters."
        self.logger.log( closeDistStr.format(bestNodeNum, bestDist) )

        if bestScore > 30:
            if bestDist <= 0.8:
                return bestNodeNum, "very confident."
            else:
                return bestNodeNum, "confident, but far away."
        else:
            guessNodes = [bestNodeNum]
            for j in range(1, len(matchLocs)):
                (nodeNum, x, y, dist) = self.olin.findClosestNode(matchLocs[j])
                if nodeNum not in guessNodes:
                    guessNodes.append(nodeNum)

            if len(guessNodes) == 1 and bestDist <= 0.8:
                return guessNodes[0], "close, but guessing."
            elif len(guessNodes) == 1:
                return guessNodes[0], "far and guessing."
            else:
                nodes = str(guessNodes[0])
                for i in range(1,len(guessNodes)):
                    nodes += " or " + str(guessNodes[i])
                # TODO: figure out if bestHead is the right result, but for now it's probably ignored anyway
                return nodes, "totally unsure."


    def setLocation(self, conf, loc):
        if conf == "very confident." or conf == "confident, but far away.":
            self.lastKnownLoc = loc
            self.confidence = 10.0
        elif conf == "close, but guessing.":
            self.lastKnownLoc = loc
            self.confidence = max(0.0, self.confidence - 0.5)
        elif conf == "far and guessing.":
            self.lastKnownLoc = loc
            self.confidence = 5.0
        elif conf == "lost":
            self.lastKnownLoc = None
            self.confidence = 0.0
        else:
            self.confidence = 0.0

    def odometer(self):
        formStr = "Odometer loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})  confidence = {3:4.2f}"
        x, y, yaw = self.robot.getOdomData()
        # (nearNode, closeX, closeY, bestVal) = self.olin.findClosestNode((x,y,yaw))
        self.logger.log( formStr.format(x, y, yaw, self.odomScore) )
        self.odomScore = max(0.001, self.odomScore - 0.75)

        if self.robot.hasWheelDrop():
            self.odomScore = 0.001

        if not self.olin.isAllowedLocation((x,y)):
            self.odomScore = 0.001

        return x, y, yaw


    # def setLastLoc(self, prevDestXY):
    #     (x, y) = prevDestXY
    #     if self.lastKnownLoc == None:
    #         self.lastKnownLoc = (x, y, 0)
    #     else:
    #         (oldX, oldY, oldH) = self.lastKnownLoc
    #         self.lastKnownLoc = (x, y, oldH)

