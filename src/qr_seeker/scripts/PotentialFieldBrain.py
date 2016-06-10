# A Potential Field control system

import ReactiveBrain
import math

#-------------------------------------------
#

class PotentialFieldBrain(ReactiveBrain.ReactiveBrain):
	""" This class represents a brain for a potential field reactive system.
	This continues to use the concept of a behavior, but potential field
	behaviors each produce a vector describing the force they compute on the
	robot. The brain does vector addition to combine those vectors, and then
	transforms the resulting force on the robot into a movement direction and
	speed"""


	#----------------------------------------
	#Initialization and destruction routines

	def __init__(self, robot, maxMagnitude = 100.0):
		"""Initializes the brain with the robot it controls. Also takes
		an optional input to define the maximum magnitude of any vector."""
		super(PotentialFieldBrain, self).__init__(robot)

		# set maximum possible magnitude
		self.maxMagnitude = maxMagnitude


	#
	def step(self):
		"""one step means figuring out the vectors for each of the behaviors, and performing
		vector addition to combine them.  Then the resulting vector is the action taken."""
		vectors = self._updateBehaviors()
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

	#	if scaledSpeed > 0:
	#		# if scaled speed is too low, set to 0.2
	#		scaledSpeed = max(0.2, scaledSpeed)
	#	elif scaledSpeed < 0:
	#		# if scaled speed is too low, set to 0.2
	#		scaledSpeed = min(-0.2, scaledSpeed)

#		print transValue, scaledSpeed

		self.myRobot.move(transValue, scaledSpeed)



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





# ----------------------------------------------------------------------
# A behavior in this model has access to the robot as an instance variable,
# so that it can access the sensors.  It also has access to the brain class
# itself, for getting things like current goals that might be stored there.
# When its update method is called, it examines the world, and determines
# a source of force on the robot, returning the force as a tuple containing
# the force's magnitude and its direction.  The direction should be in terms
# of some global angle, either set by the heading of the robot when it starts up
# (when using a purely reactive robot like PyrobotRobot or PlayerRobot), or
# using the global angle (as WorldlyRobot uses).  Note that angles are assumed
# to be in degrees.

class PotentialFieldBehavior(ReactiveBrain.ReactiveBehavior):
	"""A behavior in this model has access to the robot, so that it can
	access the sensors. When its update method is called, it examines the
	world, and determines a source of force on the robot, returning the force
	as a tuple containing the force's magnitude and its direction. The
	direction should be given assuming that zero degrees is always the
	direction the robot is facing. Note that angles are assumed to be in
	degrees."""

	# ---------------------
	def __init__(self):
		"""Initializes the motor recommendations to zero, and the flag to false.
		Its reference to the robot is set to None, the robot is specified later
		when the behavior is added to the SubsumptionBrain object."""
		super(PotentialFieldBehavior, self).__init__()
		self._magnitude = 0
		self._angle = 0




	# ---------------------
	# Accessors to avoid direct access to instance variables
	def getMagnitude(self):
		return self._magnitude
	def getAngle(self):
		return self._angle

	def getVector(self):
		return (self._magnitude, self._angle)

	def setVector(self, mag, ang):
		self._magnitude = mag
		self._angle = ang


	# ---------------------

	def update(self):
		"""Reads the sensor data and determines what force is on the robot from its
		sources, and reports that as the value.  Every behavior should always
		report a force, even if it is zero.EACH SUBCLASS SHOULD DEFINE
		ONE OF THESE METHODS TO DO ITS OWN THING."""
		return (0,0)
# end of PotentialFieldBehavior Class




