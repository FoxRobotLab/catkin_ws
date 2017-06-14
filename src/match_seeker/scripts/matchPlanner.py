#!/usr/bin/env python

""" ========================================================================
matchPlanner.py
Created: June 2017

This file borrows code from the qrPlanner.py in qr_seeker.
It manages the robot, from low-level motion control (using PotentialFieldBrain)
to higher level path planning, plus matching camera images to a database of images
in order to localize the robot.
======================================================================== """

import time
import math
# import numpy as np
import cv2
import rospy
from espeak import espeak

import turtleControl
import MovementHandler
import PotentialFieldBrain
import FieldBehaviors
# import ImageRecognizer
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
        # self.webCamWidth = 640
        # self.webCamHeight = 480
        # self.image, times = self.robot.getImage()

        self.brain = None
        self.whichBrain = ""

        self.logger = OutputLogger.OutputLogger(True, True)
        self.olinGraph =  MapGraph.readMapFile(basePath + graphMapData)
        self.moveHandle = MovementHandler.MovementHandler(self.robot, self.logger)
        self.pathLoc = PathLocation.PathLocation(self.olinGraph, self.logger)
        self.prevPath = []

        # change file names in OSPathDefine
        self.locator = Localizer.Localizer(self.robot, self.olinGraph, self.logger)

        self.ignoreSignTime = 0

        self.pub = rospy.Publisher('chatter', String, queue_size=10)


    def run(self, runtime=120):
        """Runs the program for the duration of 'runtime'"""
        self.setupNavBrain()
        timeout = time.time() + runtime
        iterationCount = 0
        destination = self.pathLoc.beginJourney()
        if destination:
            self.speak("Heading to " + str(destination))
            while time.time() < timeout and not rospy.is_shutdown():
                image = self.robot.getImage()[0]
                cv2.imshow("Turtlebot View", image)
                cv2.waitKey(20)

                if iterationCount > 20:
                    self.brain.step()

                if iterationCount % 30 == 0:
                    self.logger.log("-------------- New Match ---------------")
                    matchInfo = self.locator.findLocation(image)
                    print matchInfo
                    print self.whichBrain
                    if matchInfo == "look":
                        if self.whichBrain != "loc":
                            self.setupLocBrain()
                    else:
                        if self.whichBrain != "nav":
                            self.setupNavBrain()
                        if matchInfo is not None and matchInfo[3] == "at node":
                            self.logger.log("Found a good enough match: " + str(matchInfo[0:2]))
                            if self.respondToLocation(matchInfo[0:2]): #reached destination. ask for new destination again
                                self.speak("Destination reached")
                                self.robot.stop()
                                destination = self.pathLoc.beginJourney()
                                if not destination:
                                    break
                                self.speak("Heading to " + str(destination))
                        elif matchInfo is not None and matchInfo[3] == "check coord":
                            self.checkCoordinates(matchInfo[0:2])
                iterationCount += 1

        self.brain.stopAll()

    def setupLocBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions of the depth
        data. TODO: Figure out how to add a positive pull toward the next location?"""
        self.speak("Location Brain Activated")
        self.brain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        self.brain.add(FieldBehaviors.LookAround())
        self.whichBrain = "loc"



    def setupNavBrain(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions of the depth
        data. TODO: Figure out how to add a positive pull toward the next location?"""
        self.whichBrain = "nav"
        self.speak("Navigating Brain Activated")
        self.brain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        self.brain.add(FieldBehaviors.KeepMoving())
        self.brain.add(FieldBehaviors.BumperReact())
        self.brain.add(FieldBehaviors.CliffReact())
        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            self.brain.add(FieldBehaviors.ObstacleForce(i * widthPieces, widthPieces / 2, speedMultiplier))
        #     The way these pieces are made leads to the being slightly more responsive to its left side
        #     further investigation into this could lead to a more uniform obstacle reacting


    def respondToLocation(self, matchInfo):
        """Given information about a location that matches this one "enough". Uses the heading data from the match
        to determine how to turn to face the next node it should head to.
        Returns True if the robot has arrived at its final goal location, otherwise, False.
        First: gets the last node in the path the robot has traveled. If that node is different from the new
        match information, OR if they are the same but a long time has passed, then record that the robot
        has reached this location, determine which direction the robot should go next, and turn toward that direction."""

        assert matchInfo is not None

        self.logger.log("*******")
        self.logger.log("Responding to Location Reached")
        self.ignoreSignTime += 1  # Incrementing time counter to avoid responding to location for a while

        path = self.pathLoc.getPathTraveled()
        if not path:
            last = -1
        else:
            last = path[-1]

        if last != matchInfo[0] or self.ignoreSignTime > 50:
            self.ignoreSignTime = 0
            result = self.pathLoc.continueJourney(matchInfo)

            if result is None:
                # We have reached our destination
                self.prevPath.extend(self.pathLoc.getPathTraveled())
                print(self.prevPath)
                return True

            else:
                targetAngle, nextNode = result
                speakStr = "At node " + str(matchInfo[0]) + " Looking for node "+ str(nextNode)
                self.speak(speakStr)

                # We know where we are and need to turn
                self.moveHandle.turnToNextTarget(matchInfo[1], targetAngle)

        return False

    def checkCoordinates(self, matchInfo):
        nearNode = matchInfo[0]
        heading = matchInfo[1]
        currPath = self.pathLoc.getCurrentPath()
        tAngle = self.pathLoc.getTargetAngle()
        if currPath == None:
            return
        elif nearNode == currPath[0] or nearNode == currPath[1]:
            if abs(heading-tAngle) >= 5:
                self.moveHandle.turnToNextTarget(heading,tAngle)
                self.logger.log("Readjusting heading.")


    def speak(self, speakStr):
        espeak.set_voice("english-us", gender=2, age=10)
        espeak.synth(speakStr)  # nodeNum, nodeCoord, heading = matchInfo
        self.pub.publish(speakStr)


if __name__=="__main__":
    rospy.init_node('Planner')
    plan = MatchPlanner()
    plan.run(5000)
    # rospy.on_shutdown(plan.exit)
    rospy.spin()
