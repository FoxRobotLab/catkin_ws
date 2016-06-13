#!/usr/bin/env python

import rospy

import PotentialFieldBrain
import TurtleBot


import cv2
import numpy

import math
import random

class ColorInvestigate(PotentialFieldBrain.PotentialFieldBehavior):

    def __init__(self, multiCamShift):
	super(ColorInvestigate, self).__init__()
	self.mcs = multiCamShift
	self.framex, framey = self.mcs.getFrameDims()
	self.xWidth = self.framex / 6.5

    def update(self):
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
		self.setVector(0.2,  180.0 + turnAngle)
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

    def __init__(self, rate):
	super(RandomWander, self).__init__()
	self.iteration = 0
	self.rate = rate
	self.speed = 0
	self.heading = 0

    def update(self):
	"""wanders with a random heading"""
	if self.iteration > self.rate:
	    self.iteration = 0
	    heading = (random.random()*180)-90
	    self.speed = 0.1
	    if heading >= 0:
		self.heading = heading
	    else:
		self.heading = 360+heading
	    self.iteration += 1
	    print self.speed, self.heading
	    self.setVector(self.speed, self.heading)


class ObstacleForce(PotentialFieldBrain.PotentialFieldBehavior):

    def __init__(self, posPercent, speedMult):
	# posPercent is distance from left side of camera image, in percent of botWidth	(from 0 to 1)
	super(ObstacleForce, self).__init__()
	self.speedMult = speedMult
	self.imageWidth = 30
	self.angle = (posPercent-0.5)*60
	self.imageLeft = (640*posPercent)-(self.imageWidth/2)
	self.imageRight = self.imageLeft+self.imageWidth
	if self.imageRight > 640:
	    self.imageLeft -= self.imageRight-640
	    self.imageRight = 640
	elif self.imageLeft < 0:
	    self.imageRight += 0-self.imageLeft
	    self.imageLeft = 0



    def update(self):
	# botWidth = 640
	# botHeight = 480

	obstVals = self.robot.getDepth(self.imageLeft,
	                               240-(self.imageWidth/2),
	                               self.imageWidth,
	                               self.imageWidth)

	masked_obstVals = numpy.ma.masked_array(obstVals, obstVals==0)
	# print numpy.ma.masked_array(obstVals, obstVals==0)

	meanDistance = numpy.mean(masked_obstVals)

	#print "--------"
	#print masked_obstVals
	#print meanDistance
	#print (50/meanDistance, 180-(self.angle))

	if meanDistance < 1500:
	    if meanDistance < 500:
		meanDistance = 500
	    self.setVector(self.speedMult/meanDistance, 180-(self.angle))
	    #self.setVector(0.0, 0.0)
	    # print self.angle
	else:
	    self.setVector(0.0, 0.0)




# -----------------------------------------------------
# Run the demo using something like this:


