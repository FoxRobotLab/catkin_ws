""" -------------------------------------------------------------------
File: ReactiveBrain.py
Author: Susan Fox
Date: March 2014

This file contains a general reactive robot "Brain" class which may be used
to implement a variety of reactive architectures. All reactive brains have a
set of behaviors, and methods for updating and running those behaviors. There
is also a ReactiveBehavior class, to capture how different behaviors are similar."""


# import random

# ---------------------------------------------------------------------
# First, the ReactiveBrain class.  


class ReactiveBrain(object):
    """A class that implements a generic reactive brain.  Doesn't actually do anything,
    but serves as a shared parent class for Subsumption and PotentialField brains."""


    def __init__(self, robot):
        """set up the Brain with an empty list of behaviors, and attach
        the robot that this brain is controlling."""
        self.behaviors = []
        self.myRobot = robot


    def add(self, behavior):
        """Takes a behavior object as input, and initializes it, and
        adds it to the list"""
        behavior.setRobot(self.myRobot)
        self.behaviors.append(behavior)


    def run(self, numSteps):
        """Takes in a number of steps, and runs the that many cycles"""
        for i in range(numSteps):
            self.step()
        self.myRobot.stop()


    def stopAll(self):
        """Stops the robot from moving, and could shut down anything else that was requried"""
        self.myRobot.stop()
        self.myRobot.exit()  # ADDED for TurtleBot


    def step(self):
        """This is the key method, which must be implemented by the subclasses. This
        would run the behaviors and choose an ultimate outcome based on them."""
        pass


# ----------------------------------------------------------------------
# A generic class for a "Behavior"  

class ReactiveBehavior(object):
    """A behavior is an object that has within it a reference to the robot
    object, so that it can access the robot's sensors, and it makes
    recommendations at any given moment in time for what the robot's
    translation and rotation should be. It also has a "flag" value that is
    set to True when the behavior is applicable, and false otherwise. This
    class is a quasi-abstract parent class for the versions of behaviors for
    subsumption and potential field architectures."""

    # ---------------------
    def __init__(self):
        """Initializes the flag to false, and sets up the robot reference.
        The flag is used to determine whether or not this behavior has a
        recommendation. The robot is specified later when the behavior is
        added to a Brain object."""
        self._flag = False
        self.robot = None
        self.enabled = True


    # ---------------------
    # Accessors to avoid direct access to instance variables
    def getFlag(self):
        """Returns current flag value."""
        return self._flag


    def getRecommendation(self):
        """Used by a brain to determine what the behavior is recommending.
        Should be defined by subclasses."""
        return None


    # ---------------------
    # Mutators change the robot's instance variables

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


    # ---------------------
    def update(self):
        """Reads the sensor data and updates the current recommendation if
        this behavior is applicable. EACH SUBCLASS SHOULD DEFINE ONE OF THESE
        METHODS TO DO ITS OWN THING."""
        pass


    def enable(self):
        """Turns on the behavior. Subclass might want to override this."""
        self.enabled = True


    def disable(self):
        """Turns off the behavior. Subclass might want to override this."""
        self.enabled = False


    def isEnabled(self):
        """Returns the status, enabled or disabled."""
        return self.enabled


# end of ReactiveBehavior Class
