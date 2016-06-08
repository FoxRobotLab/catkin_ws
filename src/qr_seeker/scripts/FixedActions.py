""" ========================================================================
FixedActions.py

Created: June, 2016

This file borrows code from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view.
======================================================================== """

import cv2
import OlinGraph
import ORBrecognizer
import rospy

class FixedActions(object):

    def __init__(self, turtleBot, cameraThread):
        """Needs the turtleBot, and cameraThread objects """
        self.robot = turtleBot
        self.camera = cameraThread
        self.d2s = 0.046 # converts degrees to seconds
        self.ORBrecog = ORBrecognizer.ORBrecognizer()


    def align(self, orbInfo):
        """Positions the robot a fixed distance from a imageMatch in front of it"""
        centerX, centerY = self.ORBrecog.getFrameCenter()
        width, height = self.ORBrecog.getFrameDims()

        while True:
            while self.camera.isStalled():
                cv2.waitKey(500)

            imageMatch, (x,y), relativeArea = orbInfo
            print("orbInfo", orbInfo)



            xScore = abs(x - centerX) / float(centerX) * 1.2
            areaScore = abs(max((1 - relativeArea / 100), -1))

            rospy.sleep(1)
            self.robot.turnLeft(0.4, 3)
            print("HEY I'M TURNING.... OR AT LEAST I SHOULD BE")
            rospy.sleep(1)
            scores = [("xScore", xScore), ("areaScore", areaScore)]

            print ("scores", scores)
            bestName, bestScore = scores[0]

            for score in scores:
                name, num = score
                if num > bestScore:
                    bestName, bestScore = score

            """ If none of the scores are big enough to return any issues with the target in the drones view to avoid
             drone constantly trying to fix minute issues"""
            if bestScore < 0.3:
                print("Image Match found is:", imageMatch)
                return imageMatch

            elif bestName == "xScore":
                if x < centerX:
                    self.turnByAngle(-90)
                    print("Turn left")
                else:
                    self.turnByAngle(90)
                    print("Turn right")

            elif bestName == "areaScore":
                # If target area does not take up enough area of turtleBot's view (too far away/close-up)
                if relativeArea < 50:
                    self.robot.forward(.05, 1)
                    print("Move forward")
                else:
                    self.robot.backward(.05, 1)
                    print("Move backward")

            return


            # xDiff = x - centerX
            # # Loop conditional#
            # if abs(xDiff) < width / 10:
            #     print("Image Match found is:", imageMatch)
            #     return imageMatch
            #----------------#
            #Backing up has first priority, then turning, then advancing
            # elif abs(xDiff) > width / 10:
            #     if x < centerX:
            #         self.turnByAngle(-12)
            #     else:
            #         self.turnByAngle(13)
            #     # self.robot.backward(0.2, 0.15)
            #     print("elif of align")
            #
            # else:
            #     print("Forward")
            #     # adjustedSpeed = 0.06 - 0.04 * (relativeArea / adjustedTargetArea)
            #     print("I'm not sure what an adjusted speed is so we are just going to go super slow")
            #     self.robot.forward(.2, 0.2)


    # def findAdjustedTargetArea(self, targetArea, angle):
    #     """Calculates an appropriate new target size depending on the angle that the robot is viewing the imageMatch."""
    #     correction = 1 - angle / 90.0
    #     return targetArea * correction


    def turnToNextTarget(self, location, destination, imageMatchOrientation):
        """Given a planned path and the orientation of the imageMatch in front of the robot, turns in the direction of the following node in the path."""
        if location == destination:
            return
        wallAngle = self.robot.findAngleToWall()
        path = OlinGraph.olin.getShortestPath(location, destination)
        currentNode, nextNode = path[0], path[1]
        targetAngle = OlinGraph.olin.getAngle(currentNode, nextNode)

        #determines actual orientation given where the robot would face if it was directly
        #looking at the imageMatch (imageMatchOrientation) and the correct angle to the imageMatch
        #(wallAngle)
        actualAngle = (imageMatchOrientation - 90 + wallAngle) % 360

        angleToTurn = targetAngle - actualAngle
        if angleToTurn < -180:
            angleToTurn += 360
        elif 180 < angleToTurn:
            angleToTurn -= 360

        print "Turning from node " , str(currentNode) , " to node " , str(nextNode)
        self.turnByAngle(angleToTurn)


    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""
        if self.camera.isStalled():
            return
        print 'Turning by an angle of: ', str(angle)
        turnSec = angle * self.d2s
        # turnSec = 3
        if angle < 0:
            turnSec = abs(turnSec)
            self.robot.turnLeft(0.4, turnSec)
            print("We are turning left now by: ", turnSec)
        else:
            self.robot.turnRight(0.4, turnSec)
            print("We are turning right now by: ", turnSec)





















