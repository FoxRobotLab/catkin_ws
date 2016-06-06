""" ========================================================================
FixedActions.py

Created: June, 2016

This file borrows code from the FixedActions.py in Speedy_nav. This file
handles the actions the turtlebot completes to see the image it is looking
for in the center of its view.
======================================================================== """

import cv2
import OlinGraph
import ORBrecognizer

class FixedActions(object):

	def __init__(self, turtleBot, cameraThread):
		"""Needs the turtleBot, and cameraThread objects """
		self.robot = turtleBot
		self.camera = cameraThread
		self.d2s = 0.046 # converts degrees to seconds
		self.OR = ORBrecognizer


	def align(self, targetRelativeArea, parent):
		"""Positions the robot a fixed distance from a imageMatch in front of it"""
		# TODO: find center of image seen with ORB to find QR imageMatch below
		# centerX, centerY = self.mcs.getFrameCenter()
		# width, height = self.mcs.getFrameDims()
		self.OR.get

		while True:
			while self.camera.isStalled():
				cv2.waitKey(500)

			bumperStatus = self.robot.getBumperStatus()
			if not parent.bumperReact(bumperStatus):
				return None

			for i in xrange(5):
				cv2.waitKey(300)
				imageMatchInfo = None #TODO: Make codeInfo a reality
				if imageMatchInfo != None:
					break
			else:
				return None

			imageMatch, (x,y), relativeArea, angleToImage = imageMatchInfo
			adjustedTargetArea = self.findAdjustedTargetArea(targetRelativeArea, angleToImage)
			areaDiff = adjustedTargetArea - relativeArea
			xDiff = x - centerX

			#Loop conditional#
			if areaDiff < 0.0:
				print "Image Match found is:", imageMatch
				return imageMatch
			#----------------#

			#Backing up has first priority, then turning, then advancing
			elif abs(xDiff) > width / 10:
				if x < centerX:
					self.turnByAngle(-12)
				else:
					self.turnByAngle(13)
				self.robot.backward(0.2, 0.15)
			else:
				print "Forward"
				adjustedSpeed = 0.06 - 0.04 * (relativeArea / adjustedTargetArea)
				print adjustedSpeed
				self.robot.forward(adjustedSpeed, 0.2)

			cv2.waitKey(800)


	def findAdjustedTargetArea(self, targetArea, angle):
		"""Calculates an appropriate new target size depending on the angle that the robot is viewing the imageMatch."""
		correction = 1 - angle / 90.0
		return targetArea * correction


	def turnToNextTarget(self, location, destination, imageMatchOrientation):
		"""Given a planned path and the orientation of the imageMatch in front of the robot, turns in the direction of the following node in the path."""
		if location == destination:
			return
		wallAngle = self.robot.findAngleToWall()
		path = OlinGraph.olin.getShortestPath(location, destination)
		currentNode, nextNode = path[0], path[1]
		targetAngle = OlinGraph.olin.getAngle(currentNode, nextNode)

		#determines actual orientation given where the robot would face if it was directly
		#looking at the imageMatch (imageMatchOrientation) and the correct angle to the imageMatch
		#(wallAngle)
		actualAngle = (imageMatchOrientation - 90 + wallAngle) % 360

		angleToTurn = targetAngle - actualAngle
		if angleToTurn < -180:
			angleToTurn += 360
		elif 180 < angleToTurn:
			angleToTurn -= 360

		print "Turning from node " , str(currentNode) , " to node " , str(nextNode)
		self.turnByAngle(angleToTurn)


	def turnByAngle(self, angle):
		"""Turns the robot by the given angle, where negative is left and positive is right"""
		if self.camera.isStalled():
			return
		print 'Turning by an angle of: ', str(angle)
		turnSec = angle * self.d2s
		if angle < 0:
			turnSec = abs(turnSec)
			self.robot.turnLeft(0.4, turnSec)
		else:
			self.robot.turnRight(0.4, turnSec)





















