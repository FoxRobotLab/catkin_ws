#!/usr/bin/env python

# import rospy
# import cv2
# import math

import PotentialFieldBrain
import numpy
import random


class ColorInvestigate(PotentialFieldBrain.PotentialFieldBehavior):
    """Attempt at a Potential Field behavior to approach a color pattern, didn't work and is
    now defunct."""

    def __init__(self, multiCamShift):
        """Sets up Behavior, stores MultiCamShift."""
        super(ColorInvestigate, self).__init__()
        self.mcs = multiCamShift
        self.framex, framey = self.mcs.getFrameDims()
        self.xWidth = self.framex / 6.5

    def update(self):
        """Gets matches from MultiCamShift and chooses move based on where it is."""
        match = self.mcs.getPotentialMatches()
        if match is None:
            self.setVector(0.0, 0.0)
        else:
            # ---- print statement ----
            print match
            x, y = match
            lx, rx = x - self.xWidth, x + self.xWidth
            centerX = self.framex / 2
            turnAngle = 45.0
            if centerX < lx:
                self.setVector(0.2, 180.0 + turnAngle)
            elif rx < centerX:
                self.setVector(0.2, 180.0 - turnAngle)
            else:
                self.setVector(0.0, 0.0)


class KeepMoving(PotentialFieldBrain.PotentialFieldBehavior):
    """This is a brain-dead class that just reports a fixed magnitude and a heading that
    matches the robot's current heading"""

    def update(self):
        """set zero magnitude and current heading"""
        self.setVector(0.2, 0.0)


class RandomWander(PotentialFieldBrain.PotentialFieldBehavior):
    """Simple behavior tht wanders, turning with some randomness each time."""

    def __init__(self, rate):
        """Sets up the behavior with all rates set to zero."""
        super(RandomWander, self).__init__()
        self.iteration = 0
        self.rate = rate
        self.speed = 0
        self.heading = 0

    def update(self):
        """wanders with a random heading."""
        if self.iteration > self.rate:
            self.iteration = 0
            heading = (random.random() * 180) - 90
            self.speed = 0.1
            if heading >= 0:
                self.heading = heading
            else:
                self.heading = 360 + heading
            self.iteration += 1
            print self.speed, self.heading
            self.setVector(self.speed, self.heading)


class ObstacleForce(PotentialFieldBrain.PotentialFieldBehavior):
    """Defines a Potential Field behavior for depth image-based obstacle reactions."""

    def __init__(self, startCol, sampWid, speedMult, imWid = 640, imHgt = 480):
        """Takes in starting column and section width in the depth image to look at,
        a speed multiplier, and optionally a setting for the size of the depth
        image, width and height. Sets up the section of the depth image to use."""
        super(ObstacleForce, self).__init__()

        self.depthImWid = imWid
        self.depthImHgt = imHgt
        self.startCol = startCol
        self.sampleWidth = sampWid
        posPercent = (self.startCol + (self.sampleWidth / 2.0)) / self.depthImWid

        self.speedMult = speedMult
        self.angle = (posPercent - 0.5) * 60

        if self.startCol + self.sampleWidth >= self.depthImWid:
            self.sampleWidth = self.depthImWid - 1

        self.startRow = self.depthImHgt / 4
        self.sampleHeight = self.depthImHgt / 2
        print("========================================================")
        print("Start col", startCol)
        print("angle", self.angle)
        print("180 + angle", 180 + self.angle)
        print("180 - angle", 180 - self.angle)
        print("========================================================")



    def update(self):
        """Gets depth data in a band as wide as suggested and some height or other,
        and computes the mean of the non-zero parts, and uses that to decide how
        fast/far to turn."""

        obstVals = self.robot.getDepth(self.startCol, self.startRow,
                                       self.sampleWidth, self.sampleHeight)

        masked_obstVals = numpy.ma.masked_array(obstVals, obstVals == 0)

        meanDistance = numpy.mean(masked_obstVals)

        # print("startCol, meanDistance", self.startCol, meanDistance)

        if meanDistance < 1500:
            if meanDistance < 500:
                meanDistance = 500
            self.setVector(self.speedMult / meanDistance, 180 + self.angle)
        else:
            self.setVector(0.0, 0.0)


