#!/usr/bin/env python

import PotentialFieldBrain
import numpy
import random


class KeepMoving(PotentialFieldBrain.PotentialFieldBehavior):
    """This is a brain-dead class that just reports a fixed magnitude and a heading that
    matches the robot's current heading"""

    def update(self):
        """set zero magnitude and current heading"""
        self.setVector(0.15, 0.0)


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
            self.setVector(self.speed, self.heading)


class BumperReact(PotentialFieldBrain.PotentialFieldBehavior):
    """Reacts to the bumper being pressed by backing up and trying to turn away from the obstacle. It reports no
    vector if the bumper is not pressed. """

    def update(self):
        """set zero magnitude and current heading"""

        bumperCode = self.robot.getBumperStatus()
        if bumperCode == 2:  # Left side of bumper was hit
            self.setVector(0.4, 220)
        elif bumperCode == 1: # should be right
            self.setVector(0.4, 160)
        elif bumperCode == 3: # should be both
            self.setVector(0.4, 180)
        else:
            self.setVector(0.0, 0.0)

class CliffReact(PotentialFieldBrain.PotentialFieldBehavior):
    """Reacts to the cliff sensor going off by stopping. It reports no
        vector if the bumper is not pressed. """

    def update(self):
        cliffCode = self.robot.getCliffStatus()
        if cliffCode != False or cliffCode != 0:
            self.emergencyStop()
        else:
            self.setVector(0.0, 0.0)

class ObstacleForce(PotentialFieldBrain.PotentialFieldBehavior):
    """Defines a Potential Field behavior for depth image-based obstacle reactions."""

    def __init__(self, startCol, sampWid, speedMult, imWid = 640, imHgt = 480):
        """Takes in starting column and section botWidth in the depth image to look at,
        a speed multiplier, and optionally a setting for the size of the depth
        image, botWidth and botHeight. Sets up the section of the depth image to use."""
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


    def update(self):
        """Gets depth data in a band as wide as suggested and some botHeight or other,
        and computes the mean of the non-zero parts, and uses that to decide how
        fast/far to turn."""

        obstVals = self.robot.getDepth(self.startCol, self.startRow,
                                       self.sampleWidth, self.sampleHeight)

        masked_obstVals = numpy.ma.masked_array(obstVals, obstVals == 0)

        if numpy.ma.count(masked_obstVals) == 0:
            meanDistance = 500
        else:
            meanDistance = numpy.mean(masked_obstVals)
            if meanDistance < 500:
                meanDistance = 500

        if meanDistance < 1200:       # Changing this value will change how sensitive robot is to walls
            self.setVector(self.speedMult / meanDistance, 180 - self.angle)
        else:
            self.setVector(0.0, 0.0)


