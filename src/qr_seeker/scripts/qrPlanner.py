#!/usr/bin/env python

""" ========================================================================
qrPlanner.py
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
import ORBrecognizer
import QRrecognizer
import FieldBehaviors
import math
import PathLocation
import numpy as np


class qrPlanner(object):

    def __init__(self):
        self.robot = turtleQR.TurtleBot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape

        self.brain = self.setupPot()
        self.image, times = self.robot.getImage()
        self.orbScanner = ORBrecognizer.ORBrecognizer(self.robot)
        self.qrScanner = QRrecognizer.QRrecognizer(self.robot)

        self.moveHandle = MovementHandler.MovementHandler(self.robot, (self.fWidth, self.fHeight))
        self.pathLoc = PathLocation.PathLocation()
        self.pathTraveled = []
        self.aligned = False
        self.ignoreBrain = False

        self.rightCam = cv2.VideoCapture(0) #laptop faces right
        self.leftCam = cv2.VideoCapture(2) #other cam faces left


    def run(self,runtime = 120):
        #Runs the program for the duration of 'runtime'"""
        timeout = time.time()+runtime
        iterationCount = 0
        self.pathLoc.beginJourney()
        while time.time() < timeout and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            leftImage = self.getNextFrame(self.leftCam)
            rightImage = self.getNextFrame(self.rightCam)
            cv2.imshow("TurtleBot View", image)
            if leftImage is not None and rightImage is not None:
                cv2.namedWindow("TurtleBot View", 0)
                cv2.imshow("TurtleBot View", np.hstack([leftImage, image, rightImage]))
                cv2.resizeWindow("TurtleBot View", 200, 100)
            cv2.waitKey(20)

            # iterationCount += 1
            # if iterationCount > 50:
            #     if not self.aligned and not self.ignoreBrain:
            #         print "Potential Field Reacting"
            #         self.brain.step()
            # dImage = self.robot.getDepth()
            # dImageInt8 = dImage.astype(np.uint8)
            # cv2.imshow("Depth View", 255 - dImageInt8)
            # cv2.waitKey(20)

            whichCam = "center" #data is from kinect camera
            orbInfo = self.orbScanner.orbScan(image, whichCam)
            qrInfo = self.qrScanner.qrScan(image)
            if orbInfo is None and leftImage is not None and rightImage is not None:
                orbLeft = self.orbScanner.orbScan(leftImage, whichCam)
                orbRight = self.orbScanner.orbScan(rightImage, whichCam)
                if orbLeft is not None and orbRight is None:
                    whichCam = "left"
                    orbInfo = orbLeft
                    qrInfo = self.qrScanner.qrScan(leftImage)
                elif orbLeft is None and orbRight is not None:
                    whichCam = "right"
                    orbInfo = orbRight
                    qrInfo = self.qrScanner.qrScan(rightImage)
                #if they're both seeing a sign there's too much noise SOMEWHERE so disregard

            if orbInfo is not None:
                if self.locate(orbInfo, qrInfo, whichCam):
                    break

        self.brain.stopAll()


    def setupPot(self):
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        currBrain.add(FieldBehaviors.KeepMoving())
        currBrain.add(FieldBehaviors.BumperReact())

        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            currBrain.add(FieldBehaviors.ObstacleForce(i * widthPieces, widthPieces/2, speedMultiplier))

        return currBrain


    def locate(self, orbInfo, qrInfo, whichCam):
        """Aligns the robot with the orbInfo in front of it, determines where it is using that orbInfo
        by seeing the QR code below. Then aligns itself with the path it should take to the next node.
        Returns True if the robot has arrived at it's destination, otherwise, False."""

        print "REACHED LOCATE"
        if qrInfo is not None:
            heading, targetAngle = self.pathLoc.continueJourney(qrInfo)

            if heading is None:
                # We have reached our destination
                print(self.pathLoc.getPath())
                return True

            # We know where we are and need to turn
            self.moveHandle.turnToNextTarget(heading, targetAngle, whichCam)
            self.ignoreBrain = False
        #TODO: make sure it doesn't see the same QR code and add it to the list loads of times
            #because it's still on screen - maybe check that the one you're seeing isn't the last one
            #you saw.

        if orbInfo is None:
            self.aligned = False
        else:
            self.ignoreBrain = True
            self.aligned = self.moveHandle.align(orbInfo, whichCam)
            time.sleep(0.1)

        return False

    def getNextFrame(self, vidObj):
        ret, frame = vidObj.read()
        return frame

    def exit(self):
        pass


if __name__=="__main__":
    rospy.init_node('Planner')
    plan = qrPlanner()
    plan.run(5000)
    rospy.on_shutdown(plan.exit)
    rospy.spin()
