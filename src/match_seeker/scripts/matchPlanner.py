#!/usr/bin/env python

""" ========================================================================
matchPlanner.py
Created: June 2017

This file borrows code from the qrPlanner.py in qr_seeker.
It manages the robot, from low-level motion control (using PotentialFieldBrain)
to higher level path planning, plus matching camera images to a database of images
in order to localize the robot.
======================================================================== """

import math

import cv2
import rospy
from espeak import espeak
import numpy as np
import turtleControl
import MovementHandler
import PotentialFieldThread
import FieldBehaviors
import Localizer
import PathLocation
import OutputLogger
import OlinWorldMap
from DataPaths import basePath, graphMapData

from std_msgs.msg import String



class MatchPlanner(object):

    def __init__(self):
        self.robot = turtleControl.TurtleBot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape

        self.brain = None
        self.goalSeeker = None
        self.whichBrain = ""

        self.logger = OutputLogger.OutputLogger(True, True)
        self.olinMap = OlinWorldMap.WorldMap()
        # self.moveHandle = MovementHandler.MovementHandler(self.robot, self.logger)
        self.pathLoc = PathLocation.PathLocation(self.olinMap, self.logger)
        self.destination = None

        self.locator = Localizer.Localizer(self.robot, self.olinMap, self.logger)

        self.ignoreLocationCount = 0

        self.pub = rospy.Publisher('chatter', String, queue_size=10)


    def run(self):
        """Runs the program for the duration of 'runtime'"""
        self.setupNavBrain()
        iterationCount = 0
        ready = self.getNextGoalDestination()

        while ready and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            cv2.imshow("Turtlebot View", image)
            # cv_image = self.robot.getDepth()
            # cv_image = cv_image.astype(np.uint8)
            # im = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
            # ret, im = cv2.threshold(cv_image,1,255,cv2.THRESH_BINARY)
            # cv2.imshaow("Depth View", im)
            cv2.waitKey(20)

            # if iterationCount > 20 and self.whichBrain == "nav":
            #     self.brain.step()


            if self.whichBrain == "loc":
                self.lookAround()


            if iterationCount % 30 == 0 or self.whichBrain == "loc":
                # odomInfo = self.locator.odometer()
                # self.checkCoordinates(odomInfo)

                self.logger.log("-------------- New Match ---------------")
                status, currPose = self.locator.findLocation(image)

                if status == "continue":            #bestMatch score > 90 but lostCount < 10
                    self.goalSeeker.setGoal(None, None, None)
                    # self.logger.log("======Goal seeker off")
                elif status == "keep-going":        #LookAround found a match
                    if self.whichBrain != "nav":
                        self.speak("Navigating...")
                        self.robot.turnByAngle(35)         #turn back 35 degrees bc the behavior is faster than the matching
                        self.brain.unpause()
                        self.checkCoordinates(currPose)    #react to the location data of the match
                        self.whichBrain = "nav"
                elif status == "look":          #enter LookAround behavior
                    if self.whichBrain != "loc":
                        self.speak("Localizing...")
                        self.brain.pause()
                        self.whichBrain = "loc"
                    self.goalSeeker.setGoal(None,None,None)
                    # self.logger.log("======Goal seeker off")
                else:                                       # found a node
                    self.whichBrain = "nav"
                    self.brain.unpause()
                    if status == "at node":
                        # self.logger.log("Found a good enough match: " + str(matchInfo))
                        self.respondToLocation(currPose)
                        if self.pathLoc.atDestination(currPose[0]):
                            # reached destination. ask for new destination again. returns false if you're not at the final node
                            self.speak("Destination reached")
                            self.robot.stop()
                            ready = self.getNextGoalDestination()
                            self.goalSeeker.setGoal(None, None, None)
                            # self.logger.log("======Goal seeker off")
                        else:
                            # h = self.pathLoc.getTargetAngle()
                            # currHead = matchInfo[1]
                            # self.goalSeeker.setGoal(self.pathLoc.getCurrentPath()[1],h,currHead)
                            self.checkCoordinates(currPose)
                            # self.logger.log("=====Updating goalSeeker: " + str(self.pathLoc.getCurrentPath()[1]) + " " +
                            #                 str(h) + " " + str(currHead))
                    elif status == "check coord":
                        self.checkCoordinates(currPose)
            iterationCount += 1

        self.logger.log("Quitting...")
        self.robot.stop()
        self.brain.stop()       # was stopAll


    def getNextGoalDestination(self):
        """Gets goal from user and sets up path location tracker etc. Returns False if
        the user wants to quit."""
        self.brain.pause()
        self.destination = self._userGoalDest()
        if self.destination == 99:
            return False
        self.pathLoc.beginJourney(self.destination)
        self.speak("Heading to " + str(self.destination))
        self.brain.unpause()
        return True


    def _userGoalDest(self):
        """Asks the user for a goal destination or 99 to cause the robot to shut down."""
        while True:
            userInp = raw_input("Enter destination index (99 to quit): ")
            if userInp.isdigit():
                userNum = int(userInp)
                if self.olinMap.isValidNode(userNum) or userNum == 99:
                    return userNum


    def setupNavBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions
        of the depth data. TODO: Figure out how to add a positive pull toward the next location?"""
        if self.whichBrain != 'nav':
            self.whichBrain = "nav"
            self.speak("Navigating Brain Activated")
            self.brain = PotentialFieldThread.PotentialFieldBrain(self.robot)
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

            speakStr = "At node " + str(nearNode)
            self.speak(speakStr)



    def checkCoordinates(self, matchInfo):
        """Check the current match information to see if we should change headings. If node that is
        confidently "not close enough" is what we expect, then make sure heading is right. Otherwise,
        if it is a neighbor of one of the expected nodes, then turn to move toward that node.
        If it is further in the path, should do something, but not doing it now.
        MatchInfo format: (nearNode, location of x, y, and heading)"""
        currPath = self.pathLoc.getCurrentPath()
        if currPath is None or currPath == []:
            return

        (nearNode, currLoc) = matchInfo
        justVisitedNode = currPath[0]
        immediateGoalNode = currPath[1]
        self.logger.log("------------- Checking coordinates -----")

        # if nearNode == currPath[-1]:
        #     self.respondToLocation(matchInfo)

        if nearNode == justVisitedNode or nearNode == immediateGoalNode:
            nextNode = immediateGoalNode
            self.logger.log("Nearest node is previous node or current goal")
        elif nearNode in currPath:
            self.logger.log("Nearest node is on current path, may have missed current goal")  # TODO: What is best response here
            pathInd = currPath.index(nearNode)
            if len(currPath)>pathInd+1 and self.olinMap.calcAngle(currLoc,currPath[pathInd+1])<20:
                nextNode = currPath[pathInd+1]
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
            if type(nearNode) == str:  #when node is x or y
                nextNode = immediateGoalNode
            else:
                nextNode = nearNode

        targetAngle = self.olinMap.calcAngle(currLoc, nextNode)
        tDist = self.olinMap.straightDist2d(currLoc, nextNode)
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
            self.speak("Adjusting heading to node " + str(node))
            self.turnToNextTarget(heading, targetHeading)
            self.goalSeeker.setGoal(None, None, None)
        elif min(angle1, angle2) >= 30:
            self.turnToNextTarget(heading, targetHeading)
        else:
            # self.goalSeeker.setGoal(tDist, targetHeading, heading)
            self.goalSeeker.setGoal(None,None,None)
            formSt = "Angle1 = {0:4.2f}     Angle2 = {1:4.2f}" #: target distance = {0:4.2f}  target heading = {1:4.2f}  current heading = {2:4.2f}"
            self.logger.log( formSt.format(angle1,angle2))#.format(tDist, targetHeading, heading) )

        self.logger.log("  targetDistance = " + str(tDist))

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

        self.brain.pause()
        self.robot.turnByAngle(angleToTurn)
        self.brain.unpause()

    def speak(self, speakStr):
        """Takes in a string and "speaks" it to the base station and also to the  robot's computer."""
        espeak.set_voice("english-us", gender=2, age=10)
        espeak.synth(speakStr)  # nodeNum, nodeCoord, heading = matchInfo
        self.pub.publish(speakStr)
        self.logger.log(speakStr)


if __name__ == "__main__":
    rospy.init_node('Planner')
    plan = MatchPlanner()
    plan.run()
    # rospy.on_shutdown(plan.exit)
    rospy.spin()
