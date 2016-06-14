""" ========================================================================
MovementHandler.py

Created: June, 2016

This file borrows code from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view.
======================================================================== """

import cv2
import numpy as np

class MovementHandler(object):

    def __init__(self, bot, botDims, webCamDims):
        """Needs the turtleBot, and cameraThread objects """
        self.robot = bot
        self.d2s = 0.046 # converts degrees to seconds
        self.botWidth, self.botHeight = botDims
        self.webCamWidth, self.webCamHeight = webCamDims


    def align(self, orbInfo, camera):
        """Positions the robot a fixed distance from a imageMatch in front of it"""
        orbReturn = self.findORBContours(orbInfo, camera)
        centerX, centerY = self.getFrameCenter(camera)
        x, relativeArea = orbReturn

        if relativeArea is None:  # No ORB return
            return False

        print("orbReturn", orbReturn)

        bestName, bestScore= self.chooseScore(x, centerX, relativeArea)

        """ If none of the scores are big enough to return any issues with the target in the bots view to avoid
         bot constantly trying to fix minute issues"""

        if bestScore < 0.4:
            return True
        # If camera found sign using Kinect
        elif camera == "center":
            self.alignCameraCenter(bestName, x, centerX, relativeArea)
        # If camera found sign using camera facing left
        elif camera == "left":
            self.alignCameraLeft(bestName, x, centerX, relativeArea)
        # If camera found sign using camera facing right
        elif camera == "right":
            self.alignCameraRight(bestName, x, centerX, relativeArea)

        # Robot has done some movement in order to align but is not yet aligned fully
        return False


    def chooseScore(self, x, centerX, relativeArea):
        """Scores each change that needs to be made to figure out how robot needs to move """
        xScore = abs(x - centerX) / float(centerX) * 1.5
        areaScore = abs(max((1 - relativeArea / 100), -1))

        print("relative Area", relativeArea)

        scores = [("xScore", xScore), ("areaScore", areaScore)]

        bestName, bestScore = scores[0]
        # Picks score that is the biggest to solve most pressing issue
        for score in scores:
            name, num = score
            if num > bestScore:
                bestName, bestScore = score

        return bestName, bestScore


    def alignCameraLeft(self, bestName, x, centerX, relativeArea):
        if bestName == "xScore":
            # If target area is not centered
            if x < centerX:
                self.robot.forward(.05, 1)
                print("Sign too far right")
            else:
                self.robot.backward(.05, 1)
                print("sign too far left")
        elif bestName == "areaScore":
            # If target area does not take up enough area of turtleBot's view (too far away/close-up)
            if relativeArea < 40:
                self.leftForwardRight()
                print("move closer to sign")
            else:
                self.rightForwardLeft()
                print("move farther from sign")


    def alignCameraCenter(self, bestName, x, centerX, relativeArea):
        if bestName == "xScore":
            # If target area is not centered
            if x < centerX:
                self.turnByAngle(-8)
                print("Turn left")
            else:
                self.turnByAngle(8)
                print("Turn right")
        elif bestName == "areaScore":
            # If target area does not take up enough area of turtleBot's view (too far away/close-up)
            if relativeArea < 80:
                self.robot.forward(.05, 1)
                print("Move forward")
            else:
                self.robot.backward(.05, 1)
                print("Move backward")


    def alignCameraRight(self, bestName, x, centerX, relativeArea):
        if bestName == "xScore":
            # If target area is not centered
            if x < centerX:
                self.robot.backward(.05, 1)
                print("sign too far left")

            else:
                self.robot.forward(.05, 1)
                print("Sign too far right")
        elif bestName == "areaScore":
            # If target area does not take up enough area of turtleBot's view (too far away/close-up)
            if relativeArea < 40:
                self.rightForwardLeft()
                print("move farther from sign")
            else:
                self.leftForwardRight()
                print("move closer to sign")


    def leftForwardRight(self):
        """Handles awkward movement for adjusting for side cameras"""
        self.turnByAngle(-90)
        self.robot.forward(.05, 1)
        self.turnByAngle(90)
        self.robot.stop()


    def rightForwardLeft(self):
        """Handles awkward movement for adjusting for side cameras"""
        self.turnByAngle(90)
        self.robot.forward(.05, 1)
        self.turnByAngle(-90)
        self.robot.stop()


    def turnToNextTarget(self, heading, targetAngle, camera):
        """Given a planned path and the orientation of the imageMatch in front of the robot, turns in the
        direction of the following node in the path."""

        wallAngle = self.robot.findAngleToWall()

        #determines actual orientation given where the robot would face if it was directly
        #looking at the imageMatch (heading) and the correct angle to the imageMatch
        #(wallAngle)
        actualAngle = (heading - 90 + wallAngle) % 360

        angleToTurn = targetAngle - actualAngle
        if camera == "left":
            angleToTurn -= 90
        elif camera == "right":
            angleToTurn += 90

        print("Angle to turn: ", angleToTurn)
        if angleToTurn < -180:
            angleToTurn += 360
        elif 180 < angleToTurn:
            angleToTurn -= 360

        self.turnByAngle(angleToTurn)


    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""
        print('Turning by an angle of: ', str(angle))
        turnSec = angle * self.d2s
        if angle < 0:
            turnSec = abs(turnSec)
            self.robot.turnLeft(0.4, turnSec)
        else:
            self.robot.turnRight(0.4, turnSec)


    """Big idea: there may be multiple contours of blue areas in the image, so we need to
    find contours of the good keyPoints, taking into account that some of these may be noise."""
    def findORBContours(self, goodKeyPoints, camera):
        w, h = self.getFrameDims(camera)
        black = np.zeros((w, h), dtype='uint8')

        # ALL the keyPoints, the list of matched keyPoints within some range
        keyPoints, goodMatches = goodKeyPoints

        # For each pair of points we have between both images
        for mat in goodMatches:
            # Get the matching keyPoints for each of the images from the match struct DMatch
            img_idx = mat.queryIdx
            (x, y) = keyPoints[img_idx].pt

            # draw large-ish white circles around each of the keyPoints
            cv2.circle(black, (int(x), int(y)), 10, (255, 255, 255), -1)

        # use closing to eliminate the white spots not in "clusters" - the noise keyPoints
        kernel = np.ones((15, 15), np.uint8)
        closing = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        # find contour of the remaining white blob. There may be small noise blobs, but we're
        # pretty sure that the largest is the one we want.
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = 0
        largestContour = None

        for i in range(0, len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > maxArea:
                maxArea = area
                largestContour = contour

        if largestContour is None:
            return (0, 0), None

        M = cv2.moments(largestContour)
        imageArea = cv2.contourArea(largestContour)
        relativeArea = float(w * h) / imageArea
        cx = int(M['m10'] / M['m00'])
        # cy = int(M['m01'] / M['m00'])
        return cx, relativeArea


    def getFrameDims(self, cam):
        """Returns the the dimmensions and depth of the camera frame"""
        if cam == "center":
            return self.botWidth, self.botHeight
        else:
            return self.webCamWidth, self.webCamHeight


    def getFrameCenter(self, cam):
        """Returns the center coordinates of the camera frame"""
        if cam == "center":
            return self.botWidth / 2, self.botHeight / 2
        else:
            return self.webCamWidth / 2, self.webCamHeight / 2
















