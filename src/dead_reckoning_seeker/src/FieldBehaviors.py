#!/usr/bin/env python

import numpy
import random
import cv2


class PotentialFieldBehavior(object):
    """A behavior in this model has access to the robot (set when the behavior is added
    to the potential field controller thread, so that it can access the sensors. When its update method
    is called, it examines the world, and determines a source of force on the robot,
    returning the force as a tuple containing the force's magnitude and its direction.
    The direction should be given assuming that zero degrees is always the
    direction the robot is facing. Note that angles are assumed to be in
    degrees."""

    # ---------------------
    def __init__(self):
        """Initializes the motor recommendations to zero, and the flag to false.
        Its reference to the robot is set to None, the robot is specified later
        when the behavior is added to the SubsumptionBrain object."""
        self._flag = False
        self.robot = None
        self.enabled = True

        self._magnitude = 0
        self._angle = 0

    # ---------------------
    # Accessors to avoid direct access to instance variables

    def getMagnitude(self):
        """Returns the magnitude set by the current behavior."""
        return self._magnitude

    def getAngle(self):
        """Returns the current angle set by the behavior."""
        return self._angle

    def getVector(self):
        """Returns a tuple of magnitude and angle."""
        return (self._magnitude, self._angle)

        # Accessors to avoid direct access to instance variables

    def getFlag(self):
        """Returns current flag value."""
        return self._flag

    def setVector(self, mag, ang):
        """Sets the current vector to a given magnitude and angle."""
        self._magnitude = mag
        self._angle = ang

    def setNoFlag(self):
        """Set the behavior flag to false"""
        self._flag = False

    def setRobot(self, robot):
        """Set the robot variable to a given robot object"""
        self.robot = robot

    def setFlag(self):
        """Sets the the recommendation flag to true, to note that a
        recommendation is being given."""
        self._flag = True

    def emergencyStop(self):
        self._magnitude = "stop"

    def update(self):
        """Reads the sensor data and determines what force is on the robot from its
        sources, and reports that as the value.  Every behavior should always
        report a force, even if it is zero.EACH SUBCLASS SHOULD DEFINE
        ONE OF THESE METHODS TO DO ITS OWN THING."""
        return (0, 0)

    def enable(self):
        """Turns on the behavior. Subclass might want to override this."""
        self.enabled = True

    def disable(self):
        """Turns off the behavior. Subclass might want to override this."""
        self.enabled = False

    def isEnabled(self):
        """Returns the status, enabled or disabled."""
        return self.enabled


# end of PotentialFieldBehavior Class


class KeepMoving(PotentialFieldBehavior):
    """This is a brain-dead class that just reports a fixed magnitude and a heading that
    matches the robot's current heading"""

    def update(self):
        """set zero magnitude and current heading"""
        self.setVector(0.15, 0.0)


class RandomWander(PotentialFieldBehavior):
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





class LookAround(PotentialFieldBehavior):
    """Turns slowly so the robot can find itself and a familiar heading"""

    def __init__(self):
        super(LookAround, self).__init__()
        self.toggle = False

    def update(self):
        if self.toggle:
            self.setVector(0.02, 90)
        else:
            self.setVector(0, 90)
        self.toggle = not self.toggle


class seekGoal(PotentialFieldBehavior):

    def __init__(self):
        super(seekGoal,self).__init__()
        self.goalHeading = None
        self.goalDist = None
        self.currHeading = None

    def update(self):
        if self.goalHeading == None:
            self.setVector(0, 0)
        else:
            angle = self.goalHeading - self.currHeading
            if self.goalDist < 1.0:
                mag = 0.05
            else:
                mag = 0.09
            self.setVector(mag,angle)

    def setGoal(self,gDist,gHeading,currHeading):
        self.goalHeading = gHeading
        self.goalDist = gDist
        self.currHeading = currHeading




class BumperReact(PotentialFieldBehavior):
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


class CliffReact(PotentialFieldBehavior):
    """Reacts to the cliff sensor going off by stopping. It reports no
        vector if the bumper is not pressed. """

    def update(self):
        cliffCode = self.robot.getCliffStatus()
        if cliffCode != False or cliffCode != 0:
            self.emergencyStop()
        else:
            self.setVector(0.0, 0.0)

class ObstacleForce(PotentialFieldBehavior):
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

        # if self.startCol == 0:
        #     cv_image = obstVals.astype(numpy.uint8)
        #     im = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
        #     # ret, im = cv2.threshold(cv_image,1,255,cv2.THRESH_BINARY)
        #     cv2.imshow("depth data", im)
        #     cv2.waitKey(20)

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


