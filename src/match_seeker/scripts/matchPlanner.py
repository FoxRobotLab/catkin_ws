#!/usr/bin/env python

""" ========================================================================
matchPlanner.py
Created: June 2017

This file borrows code from the qrPlanner.py in qr_seeker.
It manages the robot, from low-level motion control (using PotentialFieldBrain)
to higher level path planning, plus matching camera images to a database of images
in order to localize the robot.

Team Summer 2019 did not make a lot of changes here, however now the GUI doesn't
always accurately reflect what is happening. Fix what is being logged to the
GUI here and in SeekerGUI2.py

Note: Do not start matchPlanner unless the robot is on the ground, otherwise the odometry
will be off.

======================================================================== """

import math

import cv2
import rospy
# from espeak import espeak
import numpy as np
import turtleControl
# import MovementHandler
import PotentialFieldThread
import FieldBehaviors
import Localizer
import PathLocation
import OutputLogger
import OlinWorldMap
import SeekerGUI2
# from DataPaths import basePath, graphMapData
import time
import LocalizerStringConstants as loc_const
import sys

from std_msgs.msg import String


class MatchPlanner(object):

    def __init__(self):

        self.robot = turtleControl.TurtleBot()
        # print("MatchPlanner: Robot ::: Pause Movement")
        self.robot.pauseMovement()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape



        self.brain = None
        self.goalSeeker = None
        self.whichBrain = ""

        # cv2.namedWindow("Turtlebot View")
        # cv2.moveWindow("Turtlebot View",820,25)
        cv2.namedWindow("MCL Display")
        cv2.moveWindow("MCL Display", 1300, 25)

        self.logger = OutputLogger.OutputLogger(True, False)

        self.olinMap = OlinWorldMap.WorldMap()
        # self.moveHandle = MovementHandler.MovementHandler(self.robot, self.logger)
        self.pathLoc = PathLocation.PathLocation(self.olinMap, self.logger)

        # use x, y, yaw instead of node, yaw because updating odometer offsets require those
        self.startX = None
        self.startY = None
        self.startYaw = None

        self.destinationNode = None

        self.locator = None

        self.ignoreLocationCount = 0

        self.gui = SeekerGUI2.SeekerGUI2(self, self.robot)
        self.gui.update()

        self.pub = rospy.Publisher('chatter', String, queue_size=10)

    def run(self):
        """Runs the program for the duration of 'runtime'"""

        iterationCount = 0
        self.setupNavBrain()
        self.brain.pause()

        start, status = self.getStartLocation()

        if start:
            nodeAndPose = int(self.olinMap.convertLocToCell((self.startX, self.startY, self.startYaw))), (self.startX, self.startY, self.startYaw)
            ready = (start and self.getNextGoalDestination())
            self.locator = Localizer.Localizer(self.robot, self.olinMap, self.logger, self.gui)
            self.robot.unpauseMovement()
        else:
            ready = False


        while ready and not rospy.is_shutdown():
            self.gui.update()
            image = self.robot.getImage()[0]
            cv2.imshow("Turtlebot View", image)
            cv_image = self.robot.getDepth()
            cv_image = cv_image.astype(np.uint8)
            im = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            ret, im = cv2.threshold(cv_image, 1, 255, cv2.THRESH_BINARY)
            cv2.imshow("Depth View", im)
            cv2.waitKey(20)

            self.brain.unpause()

            """------------------------------------------------------------------------------------------------
            Team Summer 2019 more or less made no changes here and now the behavior in this while loop is out
            of date. Start here to make changes to reactionary behavior.

            lookAround() is currently not utilized. However in some cases it may be helpful i.e. when the
            robot has decided to stare at a wall and not move because it doesn't know where it is.

            Perhaps experiment with ways to reincorporate this behavior.
            ---------------------------------------------------------------------------------------------------"""
            if self.whichBrain == "loc":
                self.lookAround()

            # TODO: check to see if we still need this
            # odomInfo = self.locator.odometer()
            # self.checkCoordinates(odomInfo)

            self.logger.log("-------------- New Match ---------------")
            time.sleep(1)

            if status == loc_const.temp_lost:  # bestMatch score < 5 but lostCount < 10
                self.goalSeeker.setGoal(None, None, None)
                # self.logger.log("======Goal seeker off")
            elif status == loc_const.keep_going:  # LookAround found a match
                if self.whichBrain != "nav":
                    # self.speak("Navigating...")
                    self.gui.navigatingMode()
                    # self.robot.turnByAngle(35)  # turn back 35 degrees bc the behavior is faster than the matching
                    self.brain.unpause()
                    self.checkCoordinates(nodeAndPose)  # react to the location data of the match
                    self.whichBrain = "nav"
            elif status == loc_const.look:  # enter LookAround behavior
                if self.whichBrain != "loc":
                    # self.speak("Localizing...")
                    self.gui.localizingMode()
                    self.brain.pause()
                    self.whichBrain = "loc"
                self.goalSeeker.setGoal(None, None, None)
                self.lookAround()
            else:  # found a node
                if self.whichBrain == "loc":
                    self.whichBrain = "nav"
                    # self.speak("Navigating...")
                    self.gui.navigatingMode()
                    self.brain.unpause()
                if status == loc_const.at_node:
                    # self.logger.log("Found a good enough match: " + str(matchInfo))
                    self.respondToLocation(nodeAndPose)

                    if self.pathLoc.atDestination(nodeAndPose[0]):
                        # reached destination. ask for new destination again. returns false if you're not at the final node
                        # self.speak("Destination reached")
                        self.robot.stop()
                        self.robot.updateOdomLocation(nodeAndPose[1][0], nodeAndPose[1][1], nodeAndPose[1][2])
                        self.locator.odomScore = 100
                        self.getStartLocation(nextDest=True)
                        ready = self.getNextGoalDestination()
                        print("matchPlanner 168 --------- "+str(ready))
                        self.goalSeeker.setGoal(None, None, None)
                        # self.logger.log("======Goal seeker off")
                    else:
                        h = self.pathLoc.getTargetAngle()
                        currHead = nodeAndPose[1][-1]  # yaw
                        self.goalSeeker.setGoal(self.pathLoc.getCurrentPath()[1], h, currHead)
                        self.checkCoordinates(nodeAndPose)
                        # self.logger.log("=====Updating goalSeeker: " + str(self.pathLoc.getCurrentPath()[1]) + " " +
                        #                 str(h) + " " + str(currHead))

                elif status == loc_const.close:
                    self.checkCoordinates(nodeAndPose)
                self.brain.unpause()

            if ready:
                status, nodeAndPose = self.locator.findLocation(image)
                print("matchPlanner.run " + str(iterationCount) + " status=" + status + " whichBrain=" + self.whichBrain)
                iterationCount += 1
        self.shutdown()

    def getStartLocation(self, nextDest=False):
        #self.brain.pause()
        if nextDest:
            self.startX, self.startY, self.startYaw = self.robot.getOdomData()
        else:
            self.startX, self.startY, self.startYaw = self._userStartLoc()
        if self.startYaw == -1 or self.startYaw is None:
            return False, None
        self.robot.updateOdomLocation(x=self.startX, y=self.startY, yaw=self.startYaw)
        # self.brain.unpause()
        return True, loc_const.at_node

    def getNextGoalDestination(self):
        """Gets goal from user and sets up path location tracker etc. Returns False if
        the user wants to quit."""
        self.brain.pause()
        self.destinationNode = self._userGoalDest()
        if self.destinationNode == -1:
            return False
        self.pathLoc.beginJourney(self.destinationNode)
        # self.speak("Heading to " + str(self.destinationNode))
        self.brain.unpause()
        return True

    def _userStartLoc(self):
        self.gui.popupStart()
        userInputLoc = self.gui.inputStartLoc()
        userInputYaw = self.gui.inputStartYaw()
        # #where it is a choice to pick node or loc pop ups
        # self.gui.askWhich()
        # userInputX = self.gui.userInputStartX
        # userInputY = self.gui.userInputStartY

        print("User input:", userInputLoc, userInputYaw)

        userLocList = userInputLoc.split()

        # node
        if len(userLocList) == 1:
            userNode = int(userLocList[0])
            print("User node:", userNode)
            if self.olinMap.isValidNode(userNode) or userNode != -1:
                userX, userY = self.olinMap._nodeToCoord(userNode)
                print ("user x, y:", userX, userY)
            else:
                return -1,-1,-1
        # x y
        elif len(userLocList) == 2:
            userX = float(userLocList[0])
            userY = float(userLocList[1])

        return userX, userY, float(userInputYaw)

    def _userGoalDest(self):
        """Asks the user for a goal destination or -1 to cause the robot to shut down."""
        # while True:
        #     userInp = raw_input("Enter destination index (-1 to quit): ")
        #     if userInp.isdigit():
        #         userNum = int(userInp)
        #         if self.olinMap.isValidNode(userNum) or userNum == -1:
        #             return userNum
        self.gui.popupDest()
        userInput = self.gui.inputDes()
        userNum = int(userInput)
        if self.olinMap.isValidNode(userNum) or userNum == -1:
            return userNum

    def setupNavBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions
        of the depth data. TODO: Figure out how to add a positive pull toward the next location?"""
        self.whichBrain = "nav"
        # self.speak("matchPlanner.setupNavBrain: Navigating Brain Activated")
        self.brain = PotentialFieldThread.PotentialFieldBrain(self.robot)
        self.brain.pause()
        self.brain.add(FieldBehaviors.KeepMoving())
        self.brain.add(FieldBehaviors.BumperReact())
        self.brain.add(FieldBehaviors.CliffReact())
        self.goalSeeker = FieldBehaviors.seekGoal()
        self.brain.add(self.goalSeeker)
        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            obstBehavior = FieldBehaviors.ObstacleForce(i * widthPieces,
                                                        widthPieces / 2,
                                                        speedMultiplier,
                                                        self.fWidth,
                                                        self.fHeight)
            self.brain.add(obstBehavior)
            # The way these pieces are made leads to the being slightly more responsive to its left side
            # further investigation into this could lead to a more uniform obstacle reacting
        self.brain.start()
        self.brain.pause()

    def respondToLocation(self, matchInfo):
        """Given information about a location that matches this one "enough". Uses the heading data from the match
        to determine how to turn to face the next node it should head to.
        Returns True if the robot has arrived at its final goal location, otherwise, False.
        First: gets the last node in the path the robot has traveled. If that node is different from the new
        match information, OR if they are the same but a long time has passed, then record that the robot
        has reached this location, determine which direction the robot should go next, and turn toward that
        direction.
        matchInfo format = [nearNode, (x, y), heading)]"""

        assert matchInfo is not None

        # self.logger.log("*******")
        # self.logger.log("Responding to Location Reached")
        self.ignoreLocationCount += 1  # Incrementing time counter to avoid responding to location for a while

        nearNode = matchInfo[0]

        if self.pathLoc.visitNewNode(nearNode) or self.ignoreLocationCount > 50:
            self.ignoreLocationCount = 0
            self.pathLoc.continueJourney(nearNode)

            # speakStr = "At node " + str(nearNode)
            # self.speak(speakStr)

    def checkCoordinates(self, localizePose):
        """Check the current match information to see if we should change headings. If node that is
        confidently "not close enough" is what we expect, then make sure heading is right. Otherwise,
        if it is a neighbor of one of the expected nodes, then turn to move toward that node.
        If it is further in the path, should do something, but not doing it now.
        MatchInfo format: (nearNode, location of x, y, and heading)"""
        currPath = self.pathLoc.getCurrentPath()
        if currPath is None or currPath == []:
            return

        (nearNode, currLoc) = localizePose
        justVisitedNode = currPath[0]
        immediateGoalNode = currPath[1]
        self.logger.log("------------- matchPlanner.checkCoordinates: Checking coordinates -----")

        # if nearNode == currPath[-1]:
        #     self.respondToLocation(localizePose)

        if nearNode == justVisitedNode or nearNode == immediateGoalNode:
            nextNode = immediateGoalNode
            self.logger.log("Nearest node is previous node or current goal")
        elif nearNode in currPath:
            self.logger.log(
                "Nearest node is on current path, may have missed current goal")  # TODO: What is best response here
            pathInd = currPath.index(nearNode)
            if len(currPath) > pathInd + 1 and self.olinMap.calcAngle(currLoc, currPath[pathInd + 1]) < 20:
                nextNode = currPath[pathInd + 1]
            else:
                nextNode = nearNode
        elif self.olinMap.areNeighbors(nearNode, immediateGoalNode):
            self.logger.log("Nearest node is adjacent to current goal but not in path")
            nextNode = immediateGoalNode
        elif self.olinMap.areNeighbors(nearNode, justVisitedNode):
            # If near node just visited, but not near next goal, and not in path already, return to just visited
            self.logger.log("Nearest node is adjacent to previous node but not in path")
            nextNode = justVisitedNode
        else:  # near a node but you don't need to be there!
            self.logger.log("Nearest node is not on/near path")
            if type(nearNode) == str:  # when node is x or y
                nextNode = immediateGoalNode
            else:
                nextNode = nearNode

        targetAngle = self.olinMap.calcAngle(currLoc, nextNode)
        tDist = self.olinMap.straightDist2d(currLoc, nextNode)
        self.gui.updateNextNode(nextNode)
        if tDist >= 1.5:
            self.turn(nextNode, currLoc[2], targetAngle, tDist)
        else:
            self.logger.log("Not Turning. TDist = " + str(tDist))

    def turn(self, node, heading, targetHeading, tDist):
        # adjust heading based on previous if statement
        targetHeading = targetHeading % 360
        angle1 = abs(heading - targetHeading)
        angle2 = 360 - angle1

        if min(angle1, angle2) >= 90:
            self.gui.updateTurnState("Turning to node " + str(node))
            # self.speak("Adjusting heading to node " + str(node))
            self.turnToNextTarget(heading, targetHeading)
            self.goalSeeker.setGoal(None, None, None)
            self.gui.endTurn()
        elif min(angle1, angle2) >= 30:
            self.gui.updateTurnState("Turning to node " + str(node))
            self.turnToNextTarget(heading, targetHeading)
            self.gui.endTurn()
        else:
            # self.goalSeeker.setGoal(tDist, targetHeading, heading)
            self.goalSeeker.setGoal(None, None, None)
            formSt = "Angle1 = {0:4.2f}     Angle2 = {1:4.2f}"  #: target distance = {0:4.2f}  target heading = {1:4.2f}  current heading = {2:4.2f}"
            self.logger.log(formSt.format(angle1, angle2))  # .format(tDist, targetHeading, heading) )

        self.logger.log("  targetDistance = " + str(tDist))
        self.gui.updateTDist(tDist)

    def lookAround(self):
        """turns. stops. waits."""
        self.robot.turnByAngle(-35)
        self.robot.stop()

    def turnToNextTarget(self, currHeading, targetAngle):
        """Takes in the currentHeading, which comes from the heading attached to the current best matching picture,
        given in global coordinates. Also takes in the target angle, also in global coordinates. This function computes
        the angle the robot should turn.
        NOTE: Could try to use depth data to modify angle if facing a wall well enough..."""

        angleToTurn = targetAngle - (currHeading % 360)
        if angleToTurn < -180:
            angleToTurn += 360
        elif 180 < angleToTurn:
            angleToTurn -= 360

        self.logger.log("Turning to next target...")
        self.logger.log("  currHeading = " + str(currHeading))
        self.logger.log("  targetAngle = " + str(targetAngle))
        self.logger.log("  angleToTurn = " + str(angleToTurn))

        self.gui.updateTurnInfo([currHeading, targetAngle, angleToTurn])
        self.gui.update()

        self.brain.pause()
        self.robot.turnByAngle(angleToTurn)
        self.brain.unpause()

    # def speak(self, speakStr):
    #     """Takes in a string and "speaks" it to the base station and also to the  robot's computer."""
    #     espeak.set_voice("english-us", gender=2, age=60)
    #     espeak.synth(speakStr)  # nodeNum, nodeCoord, heading = matchInfo
    #     self.pub.publish(speakStr)
    #     self.gui.updateMessageText(speakStr)
    #     self.logger.log(speakStr)

    def shutdown(self):
        # TODO: This doesn't actually shut down all the way
        self.logger.log("Quitting...")
        self.logger.close()
        self.robot.stop()
        self.gui.stop()
        self.brain.stop()  # was stopAll
        cv2.destroyAllWindows()
        rospy.signal_shutdown("Button pressed")
        sys.exit(0)


if __name__ == "__main__":
    rospy.init_node('Planner')
    plan = MatchPlanner()
    plan.run()
    # rospy.on_shutdown(plan.exit)
    rospy.spin()
