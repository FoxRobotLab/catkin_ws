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


class MatchPlanner(object):

    def __init__(self):
        self.robot = turtleControl.TurtleBot()
        # self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        # self.webCamWidth = 640
        # self.webCamHeight = 480
        # self.image, times = self.robot.getImage()

        self.brain = self.setupPot()

        self.logger = OutputLogger.OutputLogger(True, True)
        self.olinGraph =  MapGraph.readMapFile(basePath + graphMapData)
        self.moveHandle = MovementHandler.MovementHandler(self.robot, self.logger)
        self.pathLoc = PathLocation.PathLocation(self.olinGraph, self.logger)

        # change file names in OSPathDefine
        self.locator = Localizer.Localizer(self.robot, self.olinGraph, self.logger)

        self.ignoreSignTime = 0


    def run(self, runtime=120):
        """Runs the program for the duration of 'runtime'"""
        timeout = time.time() + runtime
        iterationCount = 0
        self.pathLoc.beginJourney()
        while time.time() < timeout and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            cv2.imshow("Turtlebot View", image)
            cv2.waitKey(20)

            if iterationCount > 20:
                self.brain.step()

            if iterationCount % 30 == 0:
                matchInfo = self.locator.findLocation(image)
                if matchInfo is not None:
                    self.logger.log("Found a good enough match: " + str(matchInfo))
                    self.ignoreSignTime += 1  # Incrementing time counter to avoid responding to location for a while
                    if self.respondToLocation(matchInfo):
                        break

            iterationCount += 1

        self.brain.stopAll()


    def setupPot(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving, BumperReact, and CliffReact behaviors, along with ObstacleForce behaviors for six regions of the depth
        data. TODO: Figure out how to add a positive pull toward the next location?"""
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        currBrain.add(FieldBehaviors.KeepMoving())
        currBrain.add(FieldBehaviors.BumperReact())
        currBrain.add(FieldBehaviors.CliffReact())

        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            currBrain.add(FieldBehaviors.ObstacleForce(i * widthPieces, widthPieces / 2, speedMultiplier))
        #     The way these pieces are made leads to the being slightly more responsive to its left side
        #     further investigation into this could lead to a more uniform obstacle reacting
        return currBrain


    def respondToLocation(self, matchInfo):
        """Given information about a location that matches this one "enough". Uses the heading data from the match
        to determine how to turn to face the next node it should head to.
        Returns True if the robot has arrived at its final goal location, otherwise, False.
        First: gets the last node in the path the robot has traveled. If that node is different from the new
        match information, OR if they are the same but a long time has passed, then record that the robot
        has reached this location, determine which direction the robot should go next, and turn toward that direction."""

        assert matchInfo is not None

        self.logger.log("---------------")
        self.logger.log("RESPONDING TO LOCATION REACHED")

        path = self.pathLoc.getPath()
        if not path:
            last = -1
        else:
            last = path[-1]
        if last != matchInfo[0] or self.ignoreSignTime > 50:
            self.ignoreSignTime = 0
            targetAngle = self.pathLoc.continueJourney(matchInfo)
            espeak.synth("At node " + str(matchInfo[0]))     # nodeNum, nodeCoord, heading = matchInfo

            if targetAngle is None:
                # We have reached our destination
                print(self.pathLoc.getPath())
                return True

            # We know where we are and need to turn
            self.moveHandle.turnToNextTarget(matchInfo[2], targetAngle)

        return False



if __name__=="__main__":
    rospy.init_node('Planner')
    plan = MatchPlanner()
    plan.run(5000)
    # rospy.on_shutdown(plan.exit)
    rospy.spin()
