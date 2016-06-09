""" ========================================================================
MovementHandler.py

Created: June, 2016

This file borrows code from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view.
======================================================================== """

import OlinGraph
import ORBrecognizer
import cv2
import numpy as np

class MovementHandler(object):

    def __init__(self, bot, dims):
        """Needs the turtleBot, and cameraThread objects """
        self.robot = bot
        self.d2s = 0.046 # converts degrees to seconds
        self.ORBrecog = ORBrecognizer.ORBrecognizer(self.robot)
        self.width, self.height = dims


    def align(self, orbInfo):
        """Positions the robot a fixed distance from a imageMatch in front of it"""
        imageMatch, (x, y), relativeArea = self.findORBContours(orbInfo)

        centerX, centerY = self.ORBrecog.getFrameCenter()

        xScore = abs(x - centerX) / float(centerX) * 1.5
        areaScore = abs(max((1 - relativeArea / 100), -1))

        scores = [("xScore", xScore), ("areaScore", areaScore)]

        bestName, bestScore = scores[0]
        # Picks score that is the highest to solve most pressing issue
        for score in scores:
            name, num = score
            if num > bestScore:
                bestName, bestScore = score

        """ If none of the scores are big enough to return any issues with the target in the drones view to avoid
         drone constantly trying to fix minute issues"""
        if bestScore < 0.4:
            return True

        elif bestName == "xScore":
            if x < centerX:
                self.turnByAngle(-8)
                print("Turn left")
            else:
                self.turnByAngle(8)
                print("Turn right")

        elif bestName == "areaScore":
            # If target area does not take up enough area of turtleBot's view (too far away/close-up)
            if relativeArea < 60:
                self.robot.forward(.05, 1)
                print("Move forward")
            else:
                self.robot.backward(.05, 1)
                print("Move backward")

        return False


    def turnToNextTarget(self, heading, targetAngle):
        """Given a planned path and the orientation of the imageMatch in front of the robot, turns in the
        direction of the following node in the path."""

        wallAngle = self.robot.findAngleToWall()

        #determines actual orientation given where the robot would face if it was directly
        #looking at the imageMatch (heading) and the correct angle to the imageMatch
        #(wallAngle)
        actualAngle = (heading - 90 + wallAngle) % 360

        angleToTurn = targetAngle - actualAngle

        print("Angle to turn: ", angleToTurn)
        if angleToTurn < -180:
            angleToTurn += 360
        elif 180 < angleToTurn:
            angleToTurn -= 360

        self.turnByAngle(angleToTurn)


    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""
        print 'Turning by an angle of: ', str(angle)
        turnSec = angle * self.d2s
        if angle < 0:
            turnSec = abs(turnSec)
            self.robot.turnLeft(0.4, turnSec)
        else:
            self.robot.turnRight(0.4, turnSec)


    """Big idea: there may be multiple contours of blue areas in the image, so we need to
    find contours of the good keypoints, taking into account that some of these may be noise."""
    def findORBContours(self, goodKeyPoints):
        black = np.zeros((self.width, self.height), dtype='uint8')

        # ALL the keypoints, the list of matched keypoints within some range
        keypoints, goodMatches = goodKeyPoints

        # For each pair of points we have between both images
        for mat in goodMatches:
            # Get the matching keypoints for each of the images from the match struct DMatch
            img_idx = mat.queryIdx
            (x, y) = keypoints[img_idx].pt

            # draw large-ish white circles around each of the keypoints
            cv2.circle(black, (int(x), int(y)), 10, (255, 255, 255), -1)

        # use closing to eliminate the white spots not in "clusters" - the noise keypoints
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        # find contour of the remaining white blob. There may be small noise blobs, but we're
        # pretty sure that the largest is the one we want.
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.contourArea(contours[0])

        maxArea = 0
        largestContour = 0

        for i in range(0, len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > maxArea:
                maxArea = area
                largestContour = contour

        M = cv2.moments(largestContour)
        imageArea = cv2.contourArea(largestContour)
        relativeArea = float(w * h) / imageArea
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # TODO: Dissappear that string
        return ("", (cx, cy), relativeArea)




















