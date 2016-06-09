""" ========================================================================
FixedActions.py

Created: June, 2016

This file borrows code from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view.
======================================================================== """

import OlinGraph
import ORBrecognizer

class FixedActions(object):

    def __init__(self, bot):
        """Needs the turtleBot, and cameraThread objects """
        self.robot = bot
        self.d2s = 0.046 # converts degrees to seconds
        self.ORBrecog = ORBrecognizer.ORBrecognizer(self.robot)


    def align(self, orbInfo):
        """Positions the robot a fixed distance from a imageMatch in front of it"""
        centerX, centerY = self.ORBrecog.getFrameCenter()
        width, height = self.ORBrecog.getFrameDims()


        imageMatch, (x,y), relativeArea = orbInfo
        #print("orbInfo", orbInfo)

        xScore = abs(x - centerX) / float(centerX) * 1.5
        areaScore = abs(max((1 - relativeArea / 100), -1))

        scores = [("xScore", xScore), ("areaScore", areaScore)]

        #print ("scores", scores)
        bestName, bestScore = scores[0]

        for score in scores:
            name, num = score
            if num > bestScore:
                bestName, bestScore = score

        #print("bestName", bestName)

        """ If none of the scores are big enough to return any issues with the target in the drones view to avoid
         drone constantly trying to fix minute issues"""
        if bestScore < 0.4:
            #print("Image Match found is:", imageMatch)
            return imageMatch

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


    def turnToNextTarget(self, location, destination, heading):
        """Given a planned path and the orientation of the imageMatch in front of the robot, turns in the
        direction of the following node in the path."""
        if location == destination:
            return

        wallAngle = self.robot.findAngleToWall()
        path = OlinGraph.olin.getShortestPath(location, destination)
        currentNode, nextNode = path[0], path[1]
        targetAngle = OlinGraph.olin.getAngle(currentNode, nextNode)

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

        print "Turning from node " , str(currentNode) , " to node " , str(nextNode)
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





















