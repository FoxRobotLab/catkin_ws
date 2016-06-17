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

        self.moveHandle = MovementHandler.MovementHandler(self.robot, (self.fWidth, self.fHeight), (self.webCamWidth,
                            self.webCamHeight))
        self.pathLoc = PathLocation.PathLocation()
        self.pathTraveled = []
        self.aligned = False
        self.ignoreBrain = False
        self.captureNum = [0, 0, 0] #center left right

        """the two webcams won't both run on Linux unless we turn down their quality ---
        mess with as necessary. Try to keep resolution as high as you can (max is 640x480).
        If it suddnely stops working it'll throw something like a VIDEOIO out of space error,
        if this happens with a configuration that worked before, try restarting the laptop."""
        self.webCamWidth = 480
        self.webCamHeight = 360
        self.rightCam = cv2.VideoCapture(1)     # both are webcams
        self.leftCam = cv2.VideoCapture(2)
        framerate = 12
        self.rightCam.set(cv2.CAP_PROP_FPS, framerate)
        self.rightCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        self.rightCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)
        self.leftCam.set(cv2.CAP_PROP_FPS, framerate)
        self.leftCam.set(cv2.CAP_PROP_FRAME_WIDTH, self.webCamWidth)
        self.leftCam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webCamHeight)

        self.ignoreSignTime = 0


    def run(self, runtime=120):
        #Runs the program for the duration of 'runtime'
        timeout = time.time() + runtime
        iterationCount = 0
        self.pathLoc.beginJourney()
        count = 0
        while time.time() < timeout and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            leftImage = self.getNextFrame(self.leftCam)
            rightImage = self.getNextFrame(self.rightCam)
            #if we have all three images, disply them in the same window
            if leftImage is not None and rightImage is not None:
                cv2.namedWindow("TurtleBot View", cv2.WINDOW_NORMAL)
                image2 = cv2.resize(image, (leftImage.shape[1], leftImage.shape[0]), None)
                cv2.imshow("TurtleBot View", np.hstack([leftImage, image2, rightImage]))
                cv2.resizeWindow("TurtleBot View", 600, 200)
            cv2.waitKey(20)

            iterationCount += 1
            if iterationCount > 20:
                if not self.aligned and not self.ignoreBrain:
                    #print "Stepping the brain"
                    self.brain.step()

            whichCam = "center"  #assume data is from kinect camera unless told otherwise
            qrInfo = self.qrScanner.qrScan(image)
            self.ignoreSignTime += 1   # Incrementing "time" to avoid reading the same sign before moving away

            if rightImage is not None and leftImage is not None and image is not None:
                print "Left cam: "
                orbLeftNone = self.orbScanner.orbScan(leftImage, 'left')
                print "Right cam: "
                orbRightNone = self.orbScanner.orbScan(rightImage, 'right')
                print "Center cam: "
                orbCenterNone = self.orbScanner.orbScan(image, 'center')

            # we didn't see a QR code from the kinect, but we have other images to check...
            if qrInfo is None and leftImage is not None and rightImage is not None:
                qrLeft = self.qrScanner.qrScan(leftImage)
                qrRight = self.qrScanner.qrScan(rightImage)
                if qrLeft is not None and qrRight is None:
                    whichCam = "left"
                    qrInfo = self.qrScanner.qrScan(leftImage)
                    print("I'm seeing a QR code from the left webcam")
                elif qrLeft is None and qrRight is not None:
                    whichCam = "right"
                    qrInfo = self.qrScanner.qrScan(rightImage)
                    print("I'm seeing a QR code from the right webcam")
                # if they're both seeing a sign there's too much noise SOMEWHERE so disregard
            if qrInfo is not None:  # saw a QR code from one of the three images
                if self.locate(qrInfo, whichCam):
                    break
            else:  # no qr code was seen, so check orb
                orbInfo = self.orbScanner.orbScan(image, whichCam)
                # didn't see a sign from the kinect, check the side images
                if orbInfo is None and leftImage is not None and rightImage is not None:
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
                if orbInfo is not None:  # the program thinks some image had a sign in it
                    self.ignoreBrain = True
                    self.aligned = self.moveHandle.align(orbInfo, whichCam)
                else:  # orb is none, so continue on as you were
                    self.ignoreBrain = False
                    self.aligned = False
            if count == 20:
                self.saveCaptures(image, 'center')
                self.saveCaptures(leftImage, 'left')
                self.saveCaptures(rightImage, 'right')
                count = 0
            count += 1
        self.brain.stopAll()

    def saveCaptures(self, image, whichCam):
        if whichCam == 'center':
            idx = 0
        elif whichCam == 'left':
            idx = 1
        else:
            idx = 2
        cv2.imwrite("/home/macalester/Desktop/githubRepositories/catkin_ws/src/qr_seeker/res/ORBcaps/" 
                    + str(self.captureNum[idx])+ whichCam + ".jpg", image)
        self.captureNum[idx] = self.captureNum[idx] + 1


    def setupPot(self):
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        currBrain.add(FieldBehaviors.KeepMoving())
        currBrain.add(FieldBehaviors.BumperReact())

        numPieces = 6
        widthPieces = int(math.floor(self.fWidth / float(numPieces)))
        speedMultiplier = 50
        for i in range(0, numPieces):
            currBrain.add(FieldBehaviors.ObstacleForce(i * widthPieces, widthPieces / 2, speedMultiplier))
        #     The way these pieces are made leads to the being slightly more responsive to its left side
        #     further investigation into this could lead to a more uniform obstacle reacting
        return currBrain


    def locate(self, qrInfo, whichCam):
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
        if qrInfo is not None and (last != qrInfo[0] or self.ignoreSignTime > 50):
            self.ignoreSignTime = 0
            heading, targetAngle = self.pathLoc.continueJourney(qrInfo)
            espeak.synth("Seen node " + str(qrInfo[0]))     # nodeNum, nodeCoord, nodeName = qrInfo

            if heading is None:
                # We have reached our destination
                print(self.pathLoc.getPath())
                return True

            # We know where we are and need to turn
            self.moveHandle.turnToNextTarget(heading, targetAngle, whichCam)
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
