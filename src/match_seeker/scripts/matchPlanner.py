#!/usr/bin/env python

""" ========================================================================
matchPlanner.py
Created: June, 2016
This file borrows code from the Planner.py in Speedy_nav. This file
uses MovementHandler.py and PotentialFieldBrain.py.
======================================================================== """

import turtleQR
import cv2
import rospy
import time
import MovementHandler
import PotentialFieldBrain
import FieldBehaviors
import ORBRecognizer
import QRRecognizer
import ImageRecognizer
import math
import PathLocation
from OSPathDefine import basePath,directory,locData
import numpy as np
from espeak import espeak


class qrPlanner(object):

    def __init__(self):
        self.robot = turtleQR.TurtleBot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        self.webCamWidth = 640
        self.webCamHeight = 480

        self.brain = self.setupPot()
        self.image, times = self.robot.getImage()
        # self.orbScanner = ORBRecognizer.ORBRecognizer(self.robot)
        # self.qrScanner = QRRecognizer.QRrecognizer(self.robot)


        self.moveHandle = MovementHandler.MovementHandler(self.robot, (self.fWidth, self.fHeight), (self.webCamWidth,
                            self.webCamHeight))
        self.pathLoc = PathLocation.PathLocation()
        self.pathTraveled = []
        self.aligned = False
        self.ignoreBrain = False

        # change file names in OSPathDefine
        self.matcher = ImageRecognizer.ImageMatcher(self.robot, logFile=True, logShell=True,
                               dir1= basePath + directory, locFile = basePath + locData,
                               baseName="frame",
                               ext="jpg", numMatches=3)

        self.matcher.setupData()

        """the two webcams won't both run on Linux unless we turn down their quality ---
        mess with as necessary. Try to keep resolution as high as you can (max is 640x480).
        If it suddnely stops working it'll throw something like a VIDEOIO out of space error,
        if this happens with a configuration that worked before, try restarting the laptop."""

        # self.rightCam = cv2.VideoCapture(2)     # both are webcams
        # self.leftCam = cv2.VideoCapture(1)
        framerate = 12
        # self.rightCam.set(cv2.CAP_PROP_FPS, framerate)
        # self.rightCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        # self.rightCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)
        # self.leftCam.set(cv2.CAP_PROP_FPS, framerate)
        # self.leftCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        # self.leftCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)

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

            iterationCount += 1
            if iterationCount > 20:
                if not self.aligned and not self.ignoreBrain:
                    # print "Stepping the brain"
                    self.brain.step()


            if iterationCount % 30 == 0:
                matchInfo = self.matcher.matchImage(image)
                print matchInfo
                if matchInfo is not None:
                    self.ignoreSignTime += 1  # Incrementing "time" to avoid reading the same sign before moving away
                    if self.locate(matchInfo):
                        break

        self.brain.stopAll()


    def setupPot(self):
        """Sets up the potential field brain with access to the robot's sensors and motors, and add the
        KeepMoving and BumperReact behaviors, along with ObstacleForce behaviors for six regions of the depth
        data."""
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


    def locate(self, matchInfo):
        """Aligns the robot with the orbInfo in front of it, determines where it is using that orbInfo
        by seeing the QR code below. Then aligns itself with the path it should take to the next node.
        Returns True if the robot has arrived at it's destination, otherwise, False."""

        print "REACHED LOCATE"
        """Check if the QR code is the same as the last one you saw. If it is, and if it's not been a
        while since you saw it, then disregard."""
        path = self.pathLoc.getPath()
        if not path:
            last = -1
        else:
            last = path[-1]
        if matchInfo is not None and (last != matchInfo[0] or self.ignoreSignTime > 50):
            self.ignoreSignTime = 0
            targetAngle = self.pathLoc.continueJourney(matchInfo)
            # espeak.synth("Seen node " + str(qrInfo[0]))     # nodeNum, nodeCoord, nodeName = qrInfo

            if targetAngle is None:
                # We have reached our destination
                print(self.pathLoc.getPath())
                return True

            # We know where we are and need to turn
            self.moveHandle.turnToNextTarget(matchInfo[2], targetAngle)
            self.ignoreBrain = False

        return False

    def getNextFrame(self, vidObj):
        ret, frame = vidObj.read()
        return frame

    # def exit(self):
    #     pass


if __name__=="__main__":
    rospy.init_node('Planner')
    plan = qrPlanner()
    plan.run(5000)
    # rospy.on_shutdown(plan.exit)
    rospy.spin()
