""" -------------------------------------------------------------------
File: SubsumptionBrain.py
Author: Susan Fox
Date: March 2014

This file contains classes that implement the subsumption architecture for use with the Scribbler robots. The
main class it the SubsumptionBrain, which keeps a set of behaviors and manages their recommendations. There
is also a SubsumptionBehavior class, which can be used by the SubsumptionBrain to manage behaviors that
the programmer wants."""

import ReactiveBrain
import random

# ---------------------------------------------------------------------
# First, the SubsumptionBrain class.  


class SubsumptionBrain(ReactiveBrain.ReactiveBrain):
    """A class that implements the Subsumption reactive architecture. It
    keeps a list of behavior objects. The higher the index in the list, the
    more important the behavior is. Wander, as the most basic behavior, will
    be in position 0. The brain includes methods for adding behaviors to the
    list. It then has a step method, which runs one cycle of sensor to action,
    and a run method that just repeats the step."""
    
    
    def step(self):
        """Runs one cycle/step of the reactive actions. It figures out the
        highest-level behavior that is currently applicable, and takes its
        recommendation for motor values"""
        behav = self._updateBehaviors()
        controlTemplate = "{0:s} is in control"
        print (controlTemplate.format(behav.__class__.__name__))
        (translate, rotate) = behav.getRecommendation()
        self.myRobot.move(translate, rotate)


    def _updateBehaviors(self):
        """A private helper method that determines the highest applicable
        behavior, and returns it, in the process updating all the higher
        behaviors. Note that lower behaviors aren't updated at all."""

        # starting at the end and working backwards,
        for b in range(len(self.behaviors) - 1, -1, -1):
            behav = self.behaviors[b]
            behav.setNoFlag()
            behav.update()
            # if it fired, return number.  Note that the zeroth behavior always fires if nothing else does
            if behav.getFlag() or b == 0:
                return behav

# end of SubsumptionBrain Class




# ----------------------------------------------------------------------
# A class for a "Behavior"  


class SubsumptionBehavior(ReactiveBrain.ReactiveBehavior):
    """A behavior is an object that has within it a reference to the robot
    object, so that it can access the robot's sensors, and it makes
    recommendations at any given moment in time for what the robot's
    translation and rotation should be. It also has a "flag" value that is
    set to True when the behavior is applicable, and false otherwise"""
    
    # ---------------------
    def __init__(self):
        """Initializes the motor recommendations to zero, and the flag to false.
        Its reference to the robot is set to None, the robot is specified later
        when the behavior is added to the SubsumptionBrain object."""
        super(SubsumptionBehavior, self).__init__()
        self._translate = 0
        self._rotate = 0


    
    # ---------------------
    # Accessors to avoid direct access to instance variables
    def getTranslate(self):
        return self._translate
    def getRotate(self):
        return self._rotate

    def getRecommendation(self):
        """Returns the recommendation. In this case, the recommendation is a
        translation speed and a rotation speed.  This was overridden."""        
        return (self._translate, self._rotate)

    # ---------------------
    # Mutators change the robot's instance variables
    
    def setSpeeds(self, trans, rot):
        self._translate = trans
        self._rotate = rot
        
        
    # ---------------------
    def update(self):
        """Reads the sensor data and updates the current recommendation if
        this behavior is applicable. EACH SUBCLASS SHOULD DEFINE ONE OF THESE
        METHODS TO DO ITS OWN THING"""
        pass
# end of SubsumptionBehavior Class



