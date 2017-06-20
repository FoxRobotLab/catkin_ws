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

import turtleControl
import MovementHandler
import PotentialFieldBrain
import FieldBehaviors
import Localizer
import PathLocation
import OutputLogger
import MapGraph
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
        self.olinGraph = MapGraph.readMapFile(basePath + graphMapData)
        self.moveHandle = MovementHandler.MovementHandler(self.robot, self.logger)
        self.pathLoc = PathLocation.PathLocation(self.olinGraph, self.logger)
        self.prevPath = []
        self.destination = None

        self.locator = Localizer.Localizer(self.robot, self.olinGraph, self.logger)

        self.ignoreSignTime = 0

        self.pub = rospy.Publisher('chatter', String, queue_size=10)


    def run(self):
        """Runs the program for the duration of 'runtime'"""
        self.setupNavBrain()
        iterationCount = 0
        ready = self.getNextGoalDestination()

        while ready and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            cv2.imshow("Turtlebot View", image)
            cv2.waitKey(20)

            if iterationCount > 20 and self.whichBrain == "nav":
                self.brain.step()

            if self.whichBrain == "loc":
                self.moveHandle.lookAround()


            if iterationCount % 30 == 0 or self.whichBrain == "loc":
                self.logger.log("-------------- New Match ---------------")
                status, matchInfo = self.locator.findLocation(image)
                if status == "continue":
                    pass
                elif status == "keep-going":
                    if self.whichBrain != "nav":
                        self.speak("Navigating...")
                    self.whichBrain = "nav"
                elif status == "look":
                    if self.whichBrain != "loc":
                        self.speak("Localizing...")
                    self.whichBrain = "loc"
                    self.goalSeeker.setGoal(None,None,None)
                else:                                       # found a node
                    self.whichBrain = "nav"
                    if status == "at node":
                        self.logger.log("Found a good enough match: " + str(matchInfo))
                        if self.respondToLocation(matchInfo):
                            # reached destination. ask for new destination again. returns false if you're not at the final node
                            self.speak("Destination reached")
                            self.robot.stop()
                            ready = self.getNextGoalDestination()
                            self.goalSeeker.setGoal(None,None,None)
                        else:
                            h = self.pathLoc.getTargetAngle()
                            self.goalSeeker.setGoal(self.pathLoc.getCurrentPath()[1],h,h)
                    elif status == "check coord":
                        self.checkCoordinates(matchInfo)
            iterationCount += 1

        self.logger.log("Quitting...")
        self.robot.stop()
        self.brain.stopAll()


    def getNextGoalDestination(self):
        """Gets goal from user and sets up path location tracker etc. Returns False if
        the user wants to quit."""
        self.destination = self._userGoalDest(self.olinGraph.getSize())
        if self.destination == 99:
            return False
        self.pathLoc.beginJourney(self.destination)
        self.speak("Heading to " + str(self.destination))
        return True


    def _userGoalDest(self, graphSize):
        """Asks the user for a goal destination or 99 to cause the robot to shut down."""
        while True:
            userInp = raw_input("Enter destination index (99 to quit): ")
            if userInp.isdigit():
                userNum = int(userInp)
                if (0 <= userNum < graphSize) or userNum == 99:
                    return userNum


    def setupLocBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions
        of the depth data. TODO: Figure out how to add a positive pull toward the next location?"""
        if self.whichBrain != "loc":
            self.whichBrain = "loc"
            self.speak("Location Brain Activated")
            self.brain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
            self.brain.add(FieldBehaviors.LookAround())



    def setupNavBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions
        of the depth data. TODO: Figure out how to add a positive pull toward the next location?"""
        if self.whichBrain != 'nav':
            self.whichBrain = "nav"
            self.speak("Navigating Brain Activated")
            self.brain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
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
            #     The way these pieces are made leads to the being slightly more responsive to its left side
            #     further investigation into this could lead to a more uniform obstacle reacting


    def respondToLocation(self, matchInfo):
        """Given information about a location that matches this one "enough". Uses the heading data from the match
        to determine how to turn to face the next node it should head to.
        Returns True if the robot has arrived at its final goal location, otherwise, False.
        First: gets the last node in the path the robot has traveled. If that node is different from the new
        match information, OR if they are the same but a long time has passed, then record that the robot
        has reached this location, determine which direction the robot should go next, and turn toward that
        direction.
        matchInfo format = [nearNode, heading, (x, y), descr]"""

        assert matchInfo is not None

        self.logger.log("*******")
        self.logger.log("Responding to Location Reached")
        self.ignoreSignTime += 1  # Incrementing time counter to avoid responding to location for a while

        path = self.pathLoc.getPathTraveled()
        nearNode = matchInfo[0]
        heading = matchInfo[1]
        if not path:
            last = -1
        else:
            last = path[-1]

        if last != nearNode or self.ignoreSignTime > 50:
            self.ignoreSignTime = 0
            result = self.pathLoc.continueJourney(matchInfo)

            if result is None:
                # We have reached our destination
                self.prevPath.extend(self.pathLoc.getPathTraveled())
                print(self.prevPath)
                return True

            else:
                targetAngle, nextNode = result
                speakStr = "At node " + str(nearNode) + " Looking for node " + str(nextNode)
                self.speak(speakStr)

                # We know where we are and need to turn
                self.moveHandle.turnToNextTarget(heading, targetAngle)

        return False


    def checkCoordinates(self, matchInfo):
        """Check the current match information to see if we should change headings. If node that is
        confidently "not close enough" is what we expect, then make sure heading is right. Otherwise,
        if it is a neighbor of one of the expected nodes, then turn to move toward that node.
        If it is further in the path, should do something, but not doing it now.
        MatchInfo format: (nearNode, heading, (x, y))"""
        currPath = self.pathLoc.getCurrentPath()
        if currPath is None:
            return

        nearNode = matchInfo[0]
        heading = matchInfo[1]
        currLoc = matchInfo[2]
        justVisitedNode = currPath[0]
        immediateGoalNode = currPath[1]

        if nearNode == justVisitedNode or nearNode == immediateGoalNode:
            tAngle = self.olinGraph.getAngle(currLoc, immediateGoalNode)
            tDist = self.olinGraph.straightDist(currLoc,immediateGoalNode)
            self.logger.log("Node is close to either previous node or current goal.")
        elif nearNode in currPath:
            self.logger.log("Node is in the current path, may have missed current goal, doing nothing for now...")
            tAngle = heading
            tDist = self.olinGraph.straightDist(currLoc,nearNode)
            # TODO: figure out a response in this case?
        elif nearNode in [x[0] for x in self.olinGraph.getNeighbors(immediateGoalNode)]:
            self.logger.log("Node is adjacent to current goal, " + str(immediateGoalNode) + "but not in path")
            self.logger.log("  Adjusting heading to move toward current goal")
            tAngle = self.olinGraph.getAngle(currLoc, immediateGoalNode)
            tDist = self.olinGraph.straightDist(currLoc, immediateGoalNode)
        elif nearNode in [x[0] for x in self.olinGraph.getNeighbors(justVisitedNode)]:
            # If near node just visited, but not near next goal, and not in path already, return to just visited
            self.logger.log("Node is adjacent to node just visited, " + str(justVisitedNode) + "but not in current goal or path")
            self.logger.log("  Adjusting heading to move toward just visited")
            tAngle = self.olinGraph.getAngle(currLoc, justVisitedNode)
            tDist = self.olinGraph.straightDist(currLoc,justVisitedNode)
        else:  # near a node but you don't need to be there!
            tAngle = heading
            tDist = self.olinGraph.straightDist(currLoc,nearNode)

        self.goalSeeker.setGoal(tDist,tAngle,heading)

        self.logger.log("tAngle is " + str(tAngle))
        # adjust heading based on previous if statement
        tAngle = tAngle % 360
        angle1 = abs(heading - tAngle)
        angle2 = 360 - angle1

        if min(angle1, angle2) >= 90:
            self.logger.log("Readjusting heading.")
            self.speak("Adjusting heading.")
            self.moveHandle.turnToNextTarget(heading, tAngle)


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
