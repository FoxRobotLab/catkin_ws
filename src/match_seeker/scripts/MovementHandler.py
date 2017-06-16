""" ========================================================================
MovementHandler.py

Created: June, 2016
Updated: June 2017

This file borrows some ideas from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view. It reacts differently depending on which
camera the bot is seeing out of.
======================================================================== """

import cv2
import numpy as np

class MovementHandler(object):
    """Handles the actions of the robot related to approaching and aligning with markers so that they can be read.
    It also handles turning from one marker to the new direction."""

    def __init__(self, bot, logWriter):
        """Needs the turtlebot and logger objects """
        self.robot = bot
        self.d2s = 0.046 # converts degrees to seconds
        self.logger = logWriter

    def lookAround(self):
        """turns. stops. waits."""
        self.turnByAngle(-35)
        self.robot.stop()


    def turnToNextTarget(self, currHeading, targetAngle):
        """Takes in the currentHeading, which comes from the heading attached to the current best matching picture,
        given in global coordinates. Also takes in the target angle, also in global coordinates. This function computes
        the angle the robot should turn.
        NOTE: Could try to use depth data to modify angle if facing a wall well enough..."""

        angleToTurn = targetAngle - (currHeading % 360)
        if angleToTurn < -180:
            angleToTurn += 360
        elif 180 < angleToTurn:
            angleToTurn -= 360

        self.logger.log("-------------------------------------------------")
        self.logger.log("Turning to next target...")
        self.logger.log("  currHeading = " + str(currHeading))
        self.logger.log("  targetAngle = " + str(targetAngle))
        self.logger.log("  angleToTurn = " + str(angleToTurn))
        self.logger.log("-------------------------------------------------")

        self.turnByAngle(angleToTurn)


    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""
        turnSec = abs(angle * self.d2s)
        if angle > 0:
            self.robot.turnLeft(0.4, turnSec)
        elif angle < 0:
            self.robot.turnRight(0.4, turnSec)
        else:
            # No need to turn, keep going
            pass









