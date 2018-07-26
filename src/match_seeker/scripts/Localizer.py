""" ========================================================================
localizer.py

It helps the robot to locate by the combination of image matching and MCL.
It holds certain data that matchPlanner.py doesn't and decides on what
the robot should do and passes it onto matchPlanner.py.

======================================================================== """

# from espeak import espeak

from DataPaths import basePath, imageDirectory, locData
import ImageDataset
import MonteCarloLocalize
from olri_classifier.olin_test import OlinTest

class Localizer(object):


    def __init__(self, bot, mapGraph, logger,gui):
        self.robot = bot
        self.olin = mapGraph
        self.logger = logger
        self.gui = gui

        self.lostCount = 0
        self.closeEnough = 0.6  # was 0.8
        self.lastKnownLoc = None
        self.confidence = 0
        self.navType = "Images"

        self.dataset = ImageDataset.ImageDataset(logger, gui, numMatches = 3)
        self.dataset.setupData(basePath + imageDirectory, basePath + locData, "frame", "jpg")

        self.mcl = MonteCarloLocalize.monteCarloLoc(self.olin)
        self.mcl.initializeParticles(250)
        # self.mcl.getOlinMap(basePath + "res/map/olinNewMap.txt")

        self.odomScore = 100.0
        self.olin_tester = OlinTest(recent_n_max=50)


    def findLocation(self, cameraIm):
        """Given the current camera image, update the robot's estimated current location, the confidence associated
        with that location. This method returns None if the robot is not at a "significant" location (one of the points
         in the olin graph) with a high confidence. If the robot IS somewhere significant and confidence is high
         enough, then the information about the location is returned so the planner can respond to it."""
        if self.lastKnownLoc is None:
            lklSt = "Last known loc: None so far   confidence = {0:4.2f}"
            self.logger.log( lklSt.format(self.confidence) )
            self.gui.updateLastKnownList([0,0,0,self.confidence])
        else:
            lklSt = "Last known loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})   confidence = {3:4.2f}"
            (x, y, h) = self.lastKnownLoc
            self.logger.log( lklSt.format(x, y, h, self.confidence) )
            self.gui.updateLastKnownList([x,y,h,self.confidence])

        odomLoc = self.odometer()

        moveInfo = self.robot.getTravelDist()

        ### Modify this line to substitute matchImage --> cnn pred
        # scores, matchLocs = self.dataset.matchImage(cameraIm, self.lastKnownLoc, self.confidence)

        # olin_test.test_turtlebot(cameraIm, recent_n_max=50)
        #OBJECT ORIENTED OLINTEST
        # TODO: what do we do about confidence and last known loc?
        currPred, bestPred = self.olin_tester.get_prediction(cameraIm, recent_n_max=50, confidence=None, last_known_loc=None)
        currPred = str(currPred)
        bestPred = str(bestPred)
        # self.olin.highlightCell(str(bestPred), color=(254, 127, 156))
        self.mcl.olinMap.highlightCell(str(bestPred), color=(254, 127, 156))

        scores, matchLocs = self.dataset.matchImage(cameraIm, bestPred, self.confidence)

        self.gui.updatePicLocs(matchLocs[0],matchLocs[1],matchLocs[2])
        self.gui.updatePicConf(scores)

        mclData = {'matchPoses': matchLocs,
                   'matchScores': scores,
                   'odomPose': odomLoc,
                   'odomScore': self.odomScore}

        comPose, var = self.mcl.mclCycle(mclData, moveInfo)
        (centerX, centerY, centerHead) = comPose
        centerStr = "CENTER OF PARTICLE MASS: ({0: 4.2f}, {1:4.2f}, {2:4.2f}), VARIANCE: ({3:4.2f})"
        self.logger.log(centerStr.format(centerX, centerY, centerHead, var))
        self.gui.updateMCLList([centerX,centerY,centerHead,var])

        if self.odomScore < 1 and var < 3.0:
            self.gui.updateMessageText("Scattering points around MCL")

        bestScore = scores[0]
        bestX, bestY, bestHead = matchLocs[0]

        odoUpdateStr = "UPDATING ODOMETRY TO: ({0:4.2f}, {1:4.2f}, {2:4.2f})"
        # was 50 for odomScore
        if self.odomScore < 30 and var < 5.0:
            self.logger.log(odoUpdateStr.format(centerX,centerY, 80.0))
            self.gui.updateOdomList([centerX,centerY,centerHead,80.0])
            self.gui.updateMessageText("Updating Odometry with MCL.")
            self.odomScore = 80
            self.robot.updateOdomLocation(centerX, centerY, centerHead)
        elif self.odomScore < 30 and bestScore >= 30 and var > 5:
            self.logger.log(odoUpdateStr.format(bestX, bestY, bestHead))
            self.gui.updateOdomList([bestX,bestY,bestHead,bestScore])
            self.gui.updateMessageText("Updating Odometry with Images")
            self.odomScore = bestScore
            self.robot.updateOdomLocation(bestX, bestY, bestHead)

        # print("====== before response", matchLocs, scores)

        if var < 5.0:
            # print("====== mcl")
            response = self.mclResponse(comPose, var)
        else:
            # print("====== match")
            response = self.matchResponse(matchLocs, scores)
        print "Localizer: locinfo", response
        status, matchInfo = response
        if (matchInfo is not None):
            nearNode, pose = matchInfo
            cellForPose = self.olin.convertLocToCell(pose)
            self.mcl.olinMap.highlightCell(int(cellForPose))
        return response


    def mclResponse(self, comPose, var):
        """
        :param comPose: location information
        :param var: variance of mcl
        :return: a string indicating its confidence level and a tuple with mclInfo
        """
        if self.navType != "MCL":
            self.navType = "MCL"
            self.gui.updateNavType(self.navType)
        (bestNodeNum, nodeX, nodeY, bestDist) = self.olin.findClosestNode(comPose)
        mclInfo = (bestNodeNum, comPose)
        self.lastKnownLoc = comPose
        if var < 5.0:
            self.confidence = 10.0
            conf = "very confident."
        elif var < 10.0:
            self.confidence = 5.0
            conf = "guessing."
        else:
            self.confidence = max(0.0, self.confidence-0.5)
            conf = "not sure."

        self.gui.updateCNode(bestNodeNum)
        self.gui.updateMatchStatus(conf)

        if bestDist <= self.closeEnough:
            return "at node", mclInfo
        else:
            return "check coord", mclInfo


    def matchResponse(self,matchLocs, scores):
        """a messy if statement chain that takes in match scores and locations and spits out arbitrary information that
        manages behavior in matchPlanner"""


        if self.navType != "Images":
            self.navType = "Images"
            self.gui.updateNavType(self.navType)

        bestScore = scores[0]
        bestLoc = matchLocs[0]

        if bestScore < 5: #changed from >90
            self.logger.log("      I have no idea where I am.     Lost Count = " + str(self.lostCount))
            self.gui.updateMatchStatus("no idea. Lost Count = " + str(self.lostCount))
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
            self.gui.updateCNode(guess)
            self.gui.updateMatchStatus(conf)

            if conf == "very confident." or conf == "close, but guessing.":
                self.mcl.scatter(bestLoc)
                self.gui.updateMessageText("Scatter around Image Match")
                return "at node", matchInfo
            elif conf == "confident, but far away.":
                return "check coord", matchInfo
            else:
                # Guessing but not close...
                return "keep-going", matchInfo


    def _guessLocation(self, bestScore, bestLoc, matchLocs):
        """
        :return: guess and a string indicating its confidence level
        """
        (bestNodeNum, nodeX, nodeY, bestDist) = self.olin.findClosestNode(bestLoc)
        closeDistStr = "      The closest node is {0:d} at {1:4.2f} meters."
        self.logger.log( closeDistStr.format(bestNodeNum, bestDist) )

        if bestScore > 30:
            if bestDist <=self.closeEnough:
                return bestNodeNum, "very confident."
            else:
                return bestNodeNum, "confident, but far away."
        else:
            guessNodes = [bestNodeNum]
            for j in range(1, len(matchLocs)):
                (nodeNum, x, y, dist) = self.olin.findClosestNode(matchLocs[j])
                if nodeNum not in guessNodes:
                    guessNodes.append(nodeNum)

            if len(guessNodes) == 1 and bestDist <= self.closeEnough:
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
        """
        :param conf: a string indicating the confidence level of a guess location
        :param loc: location data

        set the lastKnownLoc according to conf while also setting the class variable of confidence.
        """
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
        """
        :return: the odometry data from the robot
        """
        formStr = "Odometer loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})  confidence = {3:4.2f}"
        x, y, yaw = self.robot.getOdomData()
        # (nearNode, closeX, closeY, bestVal) = self.olin.findClosestNode((x,y,yaw))
        self.logger.log( formStr.format(x, y, yaw, self.odomScore) )
        self.gui.updateOdomList([x,y,yaw,self.odomScore])
        self.odomScore = max(0.001, self.odomScore - 0.75)

        if self.robot.hasWheelDrop():
            self.odomScore = 0.001

        if not self.olin.isAllowedLocation((x,y)):
            self.odomScore = 0.001

        return x, y, yaw
