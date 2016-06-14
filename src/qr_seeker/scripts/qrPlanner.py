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
from espeak import espeak


class qrPlanner(object):

    def __init__(self):
        self.robot = turtleQR.TurtleBot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape

        self.brain = self.setupPot()
        self.image, times = self.robot.getImage()
        self.orbScanner = ORBrecognizer.ORBrecognizer(self.robot)
        self.qrScanner = QRrecognizer.QRrecognizer(self.robot)

        self.webCamWidth = 480
        self.webCamHeight = 360

        self.moveHandle = MovementHandler.MovementHandler(self.robot, (self.fWidth, self.fHeight), (self.webCamWidth,
                            self.webCamHeight))
        self.pathLoc = PathLocation.PathLocation()
        self.pathTraveled = []
        self.aligned = False
        self.ignoreBrain = False

        self.rightCam = cv2.VideoCapture(1)     # both are webcams
        self.leftCam = cv2.VideoCapture(2)
        framerate = 30
        self.rightCam.set(cv2.CAP_PROP_FPS, framerate)
        self.rightCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        self.rightCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)
        self.leftCam.set(cv2.CAP_PROP_FPS, framerate)
        self.leftCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        self.leftCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)

        self.ignoreSignTime = 0


    def run(self, runtime=120):
        #Runs the program for the duration of 'runtime'"""
        timeout = time.time() + runtime
        iterationCount = 0
        self.pathLoc.beginJourney()
        while time.time() < timeout and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            leftImage = self.getNextFrame(self.leftCam)
            rightImage = self.getNextFrame(self.rightCam)
            if leftImage is not None and rightImage is not None:
                cv2.namedWindow("TurtleBot View", cv2.WINDOW_NORMAL)
                image2 = cv2.resize(image, (leftImage.shape[1], leftImage.shape[0]), None)
                cv2.imshow("TurtleBot View", np.hstack([leftImage, image2, rightImage]))
                cv2.resizeWindow("TurtleBot View", 600, 200)
            cv2.waitKey(20)

            iterationCount += 1
            if iterationCount > 20:
                if not self.aligned and not self.ignoreBrain:
                    # espeak.synth("Brain on")
                    print "Potential Field Reacting"
                    self.brain.step()

            # dImage = self.robot.getDepth()
            # dImageInt8 = dImage.astype(np.uint8)
            # cv2.imshow("Depth View", 255 - dImageInt8)
            # cv2.waitKey(20)

            whichCam = "center"  # data is from kinect camera

            qrInfo = self.qrScanner.qrScan(image)
            self.ignoreSignTime += 1  # Incrementing "time" to avoid reading the same sign before moving away

            if qrInfo is None and leftImage is not None and rightImage is not None:
                print "I'm scanning the left and right cameras for QR codes"
                cv2.imshow("left", leftImage)
                cv2.imshow("right", rightImage)
                qrLeft = self.qrScanner.qrScan(leftImage)
                qrRight = self.qrScanner.qrScan(rightImage)
                if qrLeft is not None and qrRight is None:
                    whichCam = "left"
                    qrInfo = self.qrScanner.qrScan(leftImage)
                    print("I'm seeing things from the left webcam")
                elif qrLeft is None and qrRight is not None:
                    whichCam = "right"
                    qrInfo = self.qrScanner.qrScan(rightImage)
                    print("I'm seeing things from the right webcam")
                #if they're both seeing a sign there's too much noise SOMEWHERE so disregard
            if qrInfo is not None:
                #espeak.synth(whichCam)
                if self.locate(qrInfo, whichCam):
                    break            
            else: #no qr code was seen, so check orb
                orbInfo = self.orbScanner.orbScan(image, whichCam)
                if orbInfo is None:
                    orbLeft = self.orbScanner.orbScan(leftImage,  "left")
                    orbRight = self.orbScanner.orbScan(leftImage, "right")
                    if orbLeft is not None and orbRight is None:
                        whichCam = "left"
                        orbInfo = orbLeft
                        print("I'm seeing ORB from the left webcam")
                    elif orbLeft is None and orbRight is not None:
                        whichCam = "right"
                        orbInfo = orbRight
                        print("I'm seeing ORB from the right webcam")
                if orbInfo is not None:
                    self.ignoreBrain = True
                    self.aligned = self.moveHandle.align(orbInfo, whichCam)
                else: #orb is none, so continue on as you were
                    self.ignoreBrain = False
                    self.aligned = False
        self.brain.stopAll()


    def setupPot(self):
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        currBrain.add(FieldBehaviors.KeepMoving())
        currBrain.add(FieldBehaviors.BumperReact())

        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            currBrain.add(FieldBehaviors.ObstacleForce(i * widthPieces, widthPieces / 2, speedMultiplier))

        return currBrain


    def locate(self, qrInfo, whichCam):
        """Aligns the robot with the orbInfo in front of it, determines where it is using that orbInfo
        by seeing the QR code below. Then aligns itself with the path it should take to the next node.
        Returns True if the robot has arrived at it's destination, otherwise, False."""

        print "REACHED LOCATE"
        path = self.pathLoc.getPath()
        if not path:
            last = -1
        else:
            last = path[-1]
        if qrInfo is not None and (last != qrInfo[1] or self.ignoreSignTime > 20):
            self.ignoreSignTime = 0
            heading, targetAngle = self.pathLoc.continueJourney(qrInfo)
            # nodeNum, nodeCoord, nodeName = qrInfo
            espeak.synth("Seen node " + str(qrInfo[0]))

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
