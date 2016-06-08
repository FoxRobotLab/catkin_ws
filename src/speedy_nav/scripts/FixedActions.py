import cv2
import OlinGraph

class FixedActions(object):

	def __init__(self, turtleBot, multiCamShift, cameraThread):
		"""Needs the turtleBot, multiCamshift, and cameraThread objects """
		self.robot = turtleBot
		self.mcs = multiCamShift
		self.camera = cameraThread
		self.d2s = 0.046 # converts degrees to seconds


	def align(self, targetRelativeArea, parent):
		"""Positions the robot a fixed distance from a pattern in front of it"""
		centerX, centerY = self.mcs.getFrameCenter()
		width, height = self.mcs.getFrameDims()

		while True:
			while self.camera.isStalled():
				cv2.waitKey(500)

			bumperStatus = self.robot.getBumperStatus()
			if not parent.bumperReact(bumperStatus):
				return None

			for i in xrange(5):
				cv2.waitKey(300)
				patternInfo = self.mcs.getHorzPatterns()
				if patternInfo != None:
					break
			else:
				return None

			pattern, (x,y), relativeArea, angleToPattern = patternInfo
			adjustedTargetArea = self.findAdjustedTargetArea(targetRelativeArea, angleToPattern)
			areaDiff = adjustedTargetArea - relativeArea
			xDiff = x - centerX

			#Loop conditional#
			if areaDiff < 0.0:
				print "Pattern found is:", pattern
				return pattern
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
		"""Calculates an appropriate new target size depending on the angle that the robot is viewing the pattern."""
		correction = 1 - angle / 90.0
		return targetArea * correction


	def turnToNextTarget(self, location, destination, patternOrientation):
		"""Given a planned path and the orientation of the pattern in front of the robot, turns in the direction of the following node in the path."""
		if location == destination:
			return
		wallAngle = self.robot.findAngleToWall()
		path = OlinGraph.olin.getShortestPath(location, destination)
		currentNode, nextNode = path[0], path[1]
		targetAngle = OlinGraph.olin.getAngle(currentNode, nextNode)

		#determines actual orrientation given where the robot would face if it was directly
		#looking at the pattern (patternOrientation) and the currect angle to the pattern
		#(wallAngle)
		actualAngle = (patternOrientation - 90 + wallAngle) % 360

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





















