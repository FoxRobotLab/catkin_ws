"""Contains a potential field reactive "brain" class, and a generic
potential field behavior class that are used to implement a reactive potential field system."""

import math
import threading
import time

# -------------------------------------------
#


class PotentialFieldBrain(threading.Thread):
    """ This class represents a brain for a potential field reactive system.
    This continues to use the concept of a behavior, but potential field
    behaviors each produce a vector describing the force they compute on the
    robot. The brain does vector addition to combine those vectors, and then
    transforms the resulting force on the robot into a movement direction and
    speed.
    Angles are given in degrees, increasing in a counterclockwise direction. Thus 45 degrees
    is to the left, and -45 is to the right. Note that 315 is also to the right."""


    # ----------------------------------------
    # Initialization and destruction routines

    def __init__(self, robot, maxMagnitude = 100.0):
        """Initializes the brain with the robot it controls. Also takes
        an optional input to define the maximum magnitude of any vector."""
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.behaviors = []
        self.myRobot = robot
        # set maximum possible magnitude
        self.maxMagnitude = maxMagnitude


    #
    def step(self):
        """one step means figuring out the vectors for each of the behaviors, and performing
        vector addition to combine them.  Then the resulting vector is the action taken."""
        vectors = self._updateBehaviors()

        if vectors == "STOP":
            self.myRobot.stop()
            transValue = 0
            angle = 0
        else:
            (magnitude, angle) = self._vectorAdd(vectors)

            transValue = magnitude

            # if the angle is forward-heading, translate forward, else backward
            # there is probably a more clever way to do this...

            # Convert angle so that runs from 0 to 180 counterclockwise, and from 0 to -180 clockwise
            if angle > 180:
                angle = -(360 - angle)

            # If angle is backward-looking, set negative translation speed
            if abs(angle) > 90:
                transValue = -transValue

            if abs(angle) < 0.001:
                angle = 0

            # scale rotation speed by size of angle we want
            scaledSpeed = angle / 180.0

            if transValue == 0.0:
                # don't rotate if you aren't moving at all
                scaledSpeed = 0.0

            self.myRobot.move(transValue, scaledSpeed)


    def add(self, behavior):
        """Takes a behavior object as input, and initializes it, and
        adds it to the list"""
        behavior.setRobot(self.myRobot)
        self.behaviors.append(behavior)


    def run(self):
        """Takes in a number of steps, and runs the that many cycles"""
        with self.lock:
            self.runFlag = True
            self.paused = False
            self.stopFlag = False
        runFlag = True
        paused = True
        try:
            while runFlag:
                time.sleep(0.1)
                with self.lock:
                    paused = self.paused
                    needStop = self.stopFlag
                    self.stopFlag = False
                if not paused:
                    self.step()
                if needStop:
                    self.myRobot.stop()
                with self.lock:
                    runFlag = self.runFlag
        except Exception, e:
            print e
        finally:
            self.myRobot.stop()


    def pause(self):
        with self.lock:
            self.paused = True
            self.stopFlag = True

    def stop(self):
        with self.lock:
            self.runFlag = False

    def unpause(self):
        with self.lock:
            self.paused = False

    # def stopAll(self):
    #     """Stops the robot from moving, and could shut down anything else that was requried"""
    #     self.myRobot.stop()
    #     self.myRobot.exit()  # ADDED for TurtleBot

    def _updateBehaviors(self):
        """Run through all behaviors, and ask them to calculate the force
        they detect. Return the forces as a list. Note: forces are given as a
        tuple containing magnitude and direction"""
        vectors = []
        for behav in self.behaviors:
            if not behav.isEnabled():
                continue
            behav.setNoFlag()
            behav.update()
            vec = behav.getVector()
            if vec[0] == "stop":
                return "STOP"
            vectors.append(vec)
        return vectors


    def _vectorAdd(self, vectors):
        """takes in a list of vectors and produces the final vector.  Notice
        that, for simplicity, the behaviors return a magnitude/angle description
        of a vector, but that having the vector described as an x and y offset is
        much easier, so first the values are converted ,and then added."""
        xSum = 0
        ySum = 0
        for vec in vectors:
            (mag, angle) = vec
            radAngle = math.radians(angle)
            xVal = math.cos(radAngle) * mag
            yVal = math.sin(radAngle) * mag
            xSum += xVal
            ySum += yVal
        totalMag = math.hypot(xSum, ySum)
        totalAngle = math.atan2(ySum, xSum)
        degAngle = math.degrees(totalAngle)
        while degAngle < 0:
            degAngle += 360
        while degAngle > 360:
            degAngle -= 360
        return (totalMag, degAngle)



class PotentialFieldBehavior(object):
    """A behavior in this model has access to the robot (set when the behavior is added
    to the PotentialFieldBrain, so that it can access the sensors. When its update method
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


# if __name__ == "__main__":
#     robot = turtleControl.TurtleBot()
#     brain = PotentialFieldBrain(robot)
#     brain.add(FieldBehaviors.KeepMoving())
#     brain.add(FieldBehaviors.BumperReact())
#     brain.add(FieldBehaviors.CliffReact())
#     brain.start()
#
#     brain.pause()
#     say = raw_input("unpause? ")
#     brain.unpause()
#
#     say = raw_input("stop? ")
#     brain.stop()

