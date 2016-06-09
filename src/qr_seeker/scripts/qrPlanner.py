#!/usr/bin/env python

""" ========================================================================
qrPlanner.py
Created: June, 2016
This file borrows code from the Planner.py in Speedy_nav. This file
uses FixedActions.py and PotentialFieldBrain.py.
======================================================================== """

import turtleQR
import cv2
from datetime import datetime
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
import rospy
import time
import threading
from geometry_msgs.msg import Twist
from PIL import Image
import FixedActions
import PotentialFieldBrain
import OlinGraph
import numpy as np
import UpdateCamera
import MapGraph
import ORBrecognizer
import QRrecognizer
import FieldBehaviors
import math


class qrPlanner(object):

    def __init__(self):
        self.robot = turtleQR.TurtleBot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        self.brain = self.setupPot()
        self.image, times = self.robot.getImage()
        self.orbScanner = ORBrecognizer.ORBrecognizer(self.robot)
        self.qrScanner = QRrecognizer.QRrecognizer(self.robot)

        #self.camera = UpdateCamera.UpdateCamera(self.robot)

        totalNumNodes = OlinGraph.olin._numVerts
        self.destination = 999999999
        while self.destination > totalNumNodes:
            self.destination = int(input("Enter destination index: "))

        self.fixedActs = FixedActions.FixedActions(self.robot)
        self.pathTraveled = []


    def run(self,runtime = 120):
        #Runs the program for the duration of 'runtime'"""
        timeout = time.time()+runtime
        iterationCount = 0
        #self.camera.start()
        while time.time() < timeout and not rospy.is_shutdown():
            image = self.robot.getImage()[0]
            cv2.imshow("hi", image)
            cv2.waitKey(20)

            iterationCount += 1
            if iterationCount > 100:
                #print "STEPPING THE BRAIN"
                self.brain.step()
            #else:
            #    time.sleep(.01)


            #print "I finished the waitkey in qrPlanner after hi"
            orbInfo = self.orbScanner.orbScan(image)
            qrInfo = self.qrScanner.qrScan(image)
            if orbInfo is not None:
                if self.locate(orbInfo, qrInfo):
                    break

        #self.camera.haltRun()
        #self.camera.join()

        self.brain.stopAll()


    def setupPot(self):
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        currBrain.add(FieldBehaviors.KeepMoving())

        numPieces = 6
        widthPieces = int(math.floor(self.fWidth/float(numPieces)))
        speedMultiplier = 50
        for i in range (0, numPieces):
            currBrain.add(FieldBehaviors.ObstacleForce(self.robot, i*widthPieces, widthPieces, speedMultiplier))

        return currBrain


    def locate(self, orbInfo, qrInfo):
        """Aligns the robot with the orbInfo in front of it, determines where it is using that orbInfo
        by seeing the QR code below. Then aligns itself with the path it should take to the next node.
        Returns True if the robot has arrived at it's destination, otherwise, False."""

        print "REACHED LOCATE"
        if qrInfo is not None:
            """Read things"""
            nodeNum, nodeCoord, nodeName = qrInfo
            heading = OlinGraph.olin.getMarkerInfo(nodeNum)
            if heading is None:
                heading = 0
            print("Location is ", nodeName, "with number", nodeNum, "at coordinates", nodeCoord)
            self.pathTraveled.append(nodeNum)
            print ("Path travelled so far: \n", self.pathTraveled)
            # if location is None:
            #     """If we are lost and can not be found"""
                # return False
            if nodeNum == self.destination:
                print ("Arrived at destination.")
                return True
            print ("Finished align, now starting turning to next target")
            self.fixedActs.turnToNextTarget(nodeNum, self.destination, heading)

            #TODO: make sure it doesn't see the same QR code and add it to the list loads of times
            #because it's still on screen - maybe check that the one you're seeing isn't the last one
            #you saw.

        # orbInfo = self.fixedActs.align(targetRelativeArea, self)
        if orbInfo == None:
            return False

        momentInfo = self.findORBContours(orbInfo)
        self.fixedActs.align(momentInfo)

        return False


    """Big idea: there may be multiple contours of blue areas in the image, so we need to
    find contours of the good keypoints, taking into account that some of these may be noise."""
    def findORBContours(self, goodKeyPoints):
        w, h, _ = self.image.shape
        black = np.zeros((w, h), dtype='uint8')

        #ALL the keypoints, the list of matched keypoints within some range
        keypoints, goodMatches = goodKeyPoints

        # For each pair of points we have between both images
        for mat in goodMatches:

            # Get the matching keypoints for each of the images from the match struct DMatch
            img_idx = mat.queryIdx
            (x, y) = keypoints[img_idx].pt

            #draw large-ish white circles around each of the keypoints
            cv2.circle(black, (int(x), int(y)), 10, (255, 255, 255), -1)

        #use closing to eliminate the white spots not in "clusters" - the noise keypoints
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        #find contour of the remaining white blob. There may be small noise blobs, but we're
        #pretty sure that the largest is the one we want.
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.contourArea(contours[0])

        maxArea = 0
        largestContour = 0

        for i in range (0, len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > maxArea:
                maxArea = area
                largestContour = contour

        M = cv2.moments(largestContour)
        imageArea = cv2.contourArea(largestContour)
        relativeArea = float(w*h) / imageArea
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # TODO: Dissappear that string
        return ("", (cx,cy), relativeArea)


    # def bumperReact(self, bumper_state):
    #     "After the bumper is triggered, responds accordingly and return False if robot should stop"""
    #     if bumper_state == 0:
    #         return True
    #
    #     # These codes correspond to the wheels dropping
    #     if bumper_state == 16 or bumper_state == 28:
    #         return False
    #
    #     self.robot.backward(0.2, 3)
    #     #cv2.waitKey(300)
    #
    #     # imageInfo = None
    #     # for t in xrange(5):
    #     #     imageInfo = self.ORBrecog.scanImages()
    #     #     if imageInfo != None:
    #     #         return self.locate(imageInfo)
    #     #     time.sleep(0.2)
    #
    #     # left side of the bumper was hit
    #     if bumper_state == 2:
    #         self.fixedActs.turnByAngle(45)
    #     # right side of the bumper was hit
    #     if bumper_state == 1 or bumper_state == 3:
    #         self.fixedActs.turnByAngle(-45)
    #     return True
    #
    #
    # def sideSweep(self):
    #     """Turns the robot left and right in order to check for codes at its sides. Trying to minimize need for
    #     this because this slows down the navigation process."""
    #     stepsFor180Degrees = 82
    #
    #     self.fixedActs.turnByAngle(-90)
    #     with self.robot.moveControl.lock:
    #         self.robot.moveControl.paused = True
    #     try:
    #         twist = Twist()
    #         twist.linear.x = 0.0
    #         twist.angular.z = -0.4
    #         r = rospy.Rate(10)
    #         for i in xrange(stepsFor180Degrees):
    #             self.robot.moveControl.move_pub.publish(twist)
    #             r.sleep()
    #             # imageMatch = self.ORBrecog.scanImages()
    #             # Will ensure that the robot is not at too acute an angle with the code? Necessary?
    #             imageMatch = None
    #             if imageMatch != None:
    #                 return self.locate(imageMatch)
    #     except CvBridgeError as e:
    #         print (e)
    #     finally:
    #         twist = Twist()  # default to no motion
    #         self.robot.moveControl.move_pub.publish(twist)
    #         with self.robot.moveControl.lock:
    #             self.robot.moveControl.paused = False  # start up the continuous motion loop again
    #     self.fixedActs.turnByAngle(-90)
    #     return False


    def exit(self):
        #self.camera.haltRun()
        pass


if __name__=="__main__":
    rospy.init_node('Planner')
    plan = qrPlanner()
    plan.run(5000)
    rospy.on_shutdown(plan.exit)
    rospy.spin()
