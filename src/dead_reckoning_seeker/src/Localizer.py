""" ========================================================================
localizer.py

It helps the robot to locate by the combination of image matching and MCL.
It holds certain data that matchPlanner.py doesn't and decides on what
the robot should do and passes it onto matchPlanner.py.

October 2024 - Stripped down to only use odometry

======================================================================== """


import MonteCarloLocalize
import math
import LocalizerStringConstants as loc_const

class Localizer(object):

    def __init__(self, bot, mapGraph, logger, gui):
        self.robot = bot
        self.olin = mapGraph
        self.logger = logger
        self.gui = gui
        self.lostCount = 0
        self.closeEnough = 0.7  # was 0.6
        self.lastKnownLoc = None # Will always duplicate one row of the odom table, not super useful right now
        self.confidence = 0
        self.navType = "ODOM"

        xyTuple = self.olin._nodeToCoord(int(self.gui.inputStartLoc()))
        heading = self.gui.inputStartYaw()
        initPose = (xyTuple[0], xyTuple[1], float(heading))

        self.mcl = MonteCarloLocalize.monteCarloLoc(self.olin)
        self.mcl.initializeParticles(250, point=initPose)

        self.odomScore = 100.0


    def findLocation(self):
        """Find location using mcl, currently only using odometry"""
        odomLoc = self.odometer()
        moveInfo = self.robot.getTravelDist()


        if self.lastKnownLoc is None:
            self.lastKnownLoc = odomLoc
            lklSt = "Last known loc: None so far   confidence = {0:4.2f}"
            self.logger.log(lklSt.format(self.confidence))
            self.gui.updateLastKnownList([0, 0, 0, self.confidence])
        else:
            lklSt = "Last known loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})   confidence = {3:4.2f}"
            (x, y, h) = self.lastKnownLoc
            self.logger.log(lklSt.format(x, y, h, self.confidence))
            self.gui.updateLastKnownList([x, y, h, self.confidence])


        mclData = {'matchPoses': [[0,0,0], [0,0,0], [0,0,0], [0,0,0]],
                   'matchScores': [0.0, 0.0, 0.0],
                   'odomPose': odomLoc,
                   'odomScore': self.odomScore}

        comPose, var = self.mcl.mclCycle(mclData, moveInfo)
        (centerX, centerY, centerHead) = comPose
        centerStr = "CENTER OF PARTICLE MASS: ({0: 4.2f}, {1:4.2f}, {2:4.2f}), VARIANCE: ({3:4.2f})"
        self.logger.log(centerStr.format(centerX, centerY, centerHead, var))
        self.gui.updateMCLList([centerX, centerY, centerHead, var])

        if self.odomScore < 1 and var < 3.0:
            self.gui.updateMessageText("Scattering points around MCL")


        odoUpdateStr = "UPDATING ODOMETRY TO: ({0:4.2f}, {1:4.2f}, {2:4.2f})"

        # 70 was an arbitrary choice and it seems that the odometry remains accurate for longer than it takes to count
        # down to 70. Maybe do tests to find out how long odometry stays accurate enough and change the value from there

        #if self.odomScore > 70:
        try:
            cell, x, y, bestDist = self.olin.findClosestNode(self.robot.getOdomData())
            if bestDist <= self.closeEnough and self.isClose(self.robot.getOdomData(), (x, y, 0)):   # TODO: Both parts seem to be calculating the same thing, maybe simplifyt!
                response = loc_const.at_node, (int(self.olin.convertLocToCell(self.robot.getOdomData())), self.robot.getOdomData())
            else:
                response = loc_const.close, (int(self.olin.convertLocToCell(self.robot.getOdomData())), self.robot.getOdomData())
        except TypeError: #robot is not in a valid location, so reset odometry to center of nearest cell
            cell, x,  y, _ = self.olin.findClosestNode(self.robot.getOdomData())
            # TODO: Update the confidence value for the odometry
            inbounds_loc = self.closest_bound_pt(cell, self.robot.getOdomData()[0], self.robot.getOdomData()[1])
            self.robot.updateOdomLocation(inbounds_loc[0], inbounds_loc[1], self.robot.getOdomData()[2])
            response = loc_const.close, (cell, (inbounds_loc[0], inbounds_loc[1], self.robot.getOdomData()[2]))

        self.lastKnownLoc = self.robot.getOdomData()


        return response


    def mclResponse(self, comPose, var):
        """
        Called when odometry is bad to generate a response location
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
            self.confidence = 100
            conf = loc_const.conf_close
        elif var < 10.0:
            self.confidence = 50
            conf = loc_const.conf_far_guessing
        else:
            self.confidence = max(0.0, self.confidence - 5)
            conf = loc_const.conf_none

        self.gui.updateCNode(bestNodeNum)
        self.gui.updateMatchStatus(conf)

        if bestDist <= self.closeEnough and self.isClose(self.robot.getOdomData(), (nodeX, nodeY, 0)): # TODO: Check this
            return loc_const.at_node, mclInfo
        else:
            return loc_const.close, mclInfo

    def isClose(self, odomLoc, bestLoc):
        odomX, odomY, _ = odomLoc
        bestX, bestY, _ = bestLoc
        dist = math.hypot(odomX - bestX, odomY - bestY)
        return dist <= 1 #was 5

    def _guessLocation(self, bestScore, bestLoc, matchLocs):
        """
        :return: guess and a string indicating its confidence level
        """
        (bestNodeNum, nodeX, nodeY, bestDist) = self.olin.findClosestNode(bestLoc)
        closeDistStr = "      The closest node is {0:d} at {1:4.2f} meters."
        self.logger.log(closeDistStr.format(bestNodeNum, bestDist))

        # calculates confidence for best node prediction. If best score was too low, goes on to calculate confidence for other predictions
        if bestScore > 90:  # was 30
            if bestDist <= self.closeEnough:
                return bestNodeNum, loc_const.conf_close
            else:
                return bestNodeNum, loc_const.conf_far
        else:
            guessNodes = [bestNodeNum]
            for j in range(1, len(matchLocs)):
                (nodeNum, x, y, dist) = self.olin.findClosestNode(matchLocs[j])
                if nodeNum not in guessNodes:
                    guessNodes.append(nodeNum)
            # TODO: Fix weird code below, really needs to be simplified, is this ever used??
            else:
                nodes = str(guessNodes[0])
                for i in range(1, len(guessNodes)):
                    nodes += " or " + str(guessNodes[i])
                # TODO: figure out if bestHead is the right result, but for now it's probably ignored anyway
                return nodes, loc_const.conf_none


    def setLocation(self, conf, loc):
        """
        :param conf: a string indicating the confidence level of a guess location
        :param loc: location data

        set the lastKnownLoc according to conf while also setting the class variable of confidence.
        """
        if conf == loc_const.conf_close or conf == loc_const.conf_far:
            self.lastKnownLoc = loc
            self.confidence = 100
        elif conf == loc_const.conf_close_guessing:
            self.lastKnownLoc = loc
            self.confidence = max(0.0, self.confidence - 05)
        elif conf == loc_const.conf_none:
            self.lastKnownLoc = None
            self.confidence = 0.0
        else:
            self.confidence = 0.0


    def closest_bound_pt(self, cell, robot_x, robot_y):
        print "Out of bounds!"

        cornerpts = self.olin.cellData[cell]
        #add/subtract 0.2 to nudge the robot back in bounds
        x1 = cornerpts[0] + 0.2
        x2 = cornerpts[2] - 0.2
        y1 = cornerpts[1] + 0.2
        y2 = cornerpts[3] - 0.2
        bounding_pts = [(x1, y1), (x1, y1 + .25 * (y2 - y1)), (x1, y1 + .5 * (y2 - y1)), (x1, y1 + .75 * (y2 - y1)),
                        (x1, y2), (x1 + .25 * (x2 - x1), y2), (x1 + .5 * (x2 - x1), y2), (x1 + .75 * (x2 - x1), y2),
                        (x2, y2), (x1 + .25 * (x2 - x1), y1), (x1 + .5 * (x2 - x1), y1), (x1 + .75 * (x2 - x1), y1),
                        (x2, y1), (x2, y1 + .25 * (y2 - y1)), (x2, y1 + .5 * (y2 - y1)), (x2, y1 + .75 * (y2 - y1))]
        xr = robot_x
        yr = robot_y
        best_dist = 99
        best_pt = None
        for pt in bounding_pts:
            x1 = pt[0]
            y1 = pt[1]
            dist = math.sqrt((x1 - xr)**2 + (y1 - yr)**2)
            if dist < best_dist:
                best_dist = dist
                best_pt = (x1, y1)
        return best_pt


    def odometer(self):
        """
        :return: the odometry data from the robot
        """
        formStr = "Odometer loc: ({0:4.2f}, {1:4.2f}, {2:4.2f})  confidence = {3:4.2f}"
        x, y, yaw = self.robot.getOdomData()
        self.logger.log(formStr.format(x, y, yaw, self.odomScore))
        self.gui.updateOdomList([x, y, yaw, self.odomScore])
        self.odomScore = max(0.01, self.odomScore - 0.1)

        if self.robot.hasWheelDrop():
            self.odomScore = 0.01

        if not self.olin.isAllowedLocation((x, y)):
            self.odomScore = max(0.01, self.odomScore - 4) # TODO: test different out-of-bounds penalties
            cell, x, y, _ = self.olin.findClosestNode(self.robot.getOdomData())
            inbounds_loc = self.closest_bound_pt(cell, self.robot.getOdomData()[0], self.robot.getOdomData()[1])

            self.robot.updateOdomLocation(inbounds_loc[0], inbounds_loc[1], self.robot.getOdomData()[2])
            x = inbounds_loc[0]
            y = inbounds_loc[1]

        return x, y, yaw
