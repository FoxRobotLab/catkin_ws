#!/usr/bin/env python

import FieldBehaviors
import PotentialFieldBrain
import FixedActions
import TurtleBot
import MultiCamShift as MCS
import cv2
from datetime import datetime
from collections import deque
import OlinGraph
from cv_bridge import CvBridge, CvBridgeError
import rospy
import time

import threading


from geometry_msgs.msg import Twist


class UpdateCamera( threading.Thread ):

	def __init__(self, bot, mcs):
		threading.Thread.__init__(self)
		self.lock = threading.Lock()
		self.runFlag = True
		self.robot = bot
		self.mcs = mcs
		self.stalled = False
		self.frameAverageStallThreshold = 20

	def run(self):
		runFlag = True
		cv2.namedWindow("TurtleCam", 1)
		timesImageServed = 1
		while(runFlag):

			image, timesImageServed = self.robot.getImage()

			with self.lock:
				if timesImageServed > 20:
					if self.stalled == False:
						print "Camera Stalled!"
					self.stalled = True
				else:
					self.stalled = False
					
			
			frame = self.mcs.update(image.copy())
			cv2.imshow("TurtleCam", frame)

			code = chr(cv2.waitKey(10) & 255)

			if code == 't':
				cv2.imwrite("/home/macalester/catkin_ws/src/speedy_nav/res/captures/cap-" 
					+ str(datetime.now()) + ".jpg", image)
				print "Image saved!"
			if code == 'q':
				break

			with self.lock:
				runFlag = self.runFlag

	def isStalled(self):
		with self.lock:
			stalled = self.stalled
		return stalled

	def haltRun(self):
		with self.lock:
			self.runFlag = False




class Planner(object):

	def __init__(self):
		self.robot = TurtleBot.TurtleBot()
		self.brain = self.setupPot()		

		image, timesImageServed = self.robot.getImage()
		self.mcs = MCS.MultiCamShift(image)
		self.colorSearching = True

		self.camera = UpdateCamera(self.robot, self.mcs)

		self.destination = 999999999
		while self.destination > len(OlinGraph.nodeLocs):
			self.destination = int(input("Enter destination index: "))

		self.fixedActs = FixedActions.FixedActions(self.robot, self.mcs, self.camera)


		self.colorInvestigator = FieldBehaviors.ColorInvestigate(self.mcs)
		#self.brain.add( self.colorInvestigator )	
		self.brain.add( FieldBehaviors.KeepMoving() )

		self.obsForceList = []
		for i in range(10, 100, 10):
			behave = FieldBehaviors.ObstacleForce(i/100.0, 50)
			self.brain.add( behave )
			self.obsForceList.append(behave)

		self.pathTraveled = []





	def run(self,runtime = 120):
		"""Runs the pattern location program for the duration of 'runtime'"""
		timeout = time.time()+runtime
		self.camera.start()

		timeToWaitAfterStall = 30

		# iterators
		iterationCount = 0
		ignoreColorTime = 0
		sweepTime = 0
		sinceLastStall = 31

		self.colorInvestigator.disable()

		while time.time() < timeout and not rospy.is_shutdown():
			#rospy.sleep(0.1)
	

			if not self.camera.isStalled():
				sinceLastStall += 1
				if 30 < sinceLastStall:
					bumper = self.robot.getBumperStatus()
					if bumper != 0:
						self.fixedActs.bumperRecovery(bumper)
						self.stopColorSearching()
					
					iterationCount += 1
#					print "======================================", '\titer: ', iterationCount  
					self.brain.step()

					if not self.colorInvestigator.isEnabled():
						if sweepTime < 5000:
							sweepTime += 1
						else:
							self.sideSweep()
							print "Sweeping now"
							sweepTime = 0
					else:
						sweepTime = 0
	
					print "Colorsearching =", self.colorSearching
					print "sweepTime =", sweepTime
					
					if self.colorSearching:
						patternInfo = self.mcs.getHorzPatterns()
						if patternInfo != None and abs(patternInfo[3]) < 45:
							sweepTime = 0
							pattern, coords, relativeArea, angle = patternInfo
							node, orientationToPattern, targetRelativeArea = OlinGraph.patternLocations.get(pattern, (None, None, None))
							if node != None:
								correction = (90 - abs(angle)) / 90
								adjustedTargetArea = max(correction * targetRelativeArea, 0.07)
								if abs(relativeArea - adjustedTargetArea) > 0.07:
									self.colorInvestigator.enable()
									self.stopObstacleForce()
								else:
									if node == self.destination:
										print "Arrived at destination"
										break
									path = OlinGraph.olin.getShortestPath(node, self.destination)
									self.fixedActs.turnToNextTarget(path, orientationToPattern)
									cv2.waitKey(200)
						else:
							self.colorInvestigator.disable()
							self.startObstacleForce()
					else:
						if ignoreColorTime < 1000:
							ignoreColorTime += 1
						else:
							self.startColorSearching()
							ignoreColorTime = 0
			else:
				sinceLastStall = 0
							




#					if self.colorSearching:
#						pattern = self.mcs.getHorzPatterns()
#						# Prevents the robot from coming at the pattern from too sharp an angle
#						if pattern != None and abs(pattern[3]) < 45:
#							sweepTime = 0
#							if self.locate(pattern):
#								break
#					else:
#						if ignoreColorTime < 1000:
#							ignoreColorTime += 1
#						else:
#							self.startColorSearching()
#							ignoreColorTime = 0
#			else:
#				sinceLastStall = 0

			
		self.camera.haltRun()
		self.camera.join()
		self.brain.stopAll()
	

					
	def setupPot(self):
		"""Helpful function takes optional robot code (the six-digit Fluke board number). If code
		is given, then this connects to the robot. Otherwise, it connects to a simulated robot, and
		then creates a SubsumptionBrain object and returns it."""
		currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
		return currBrain


	def locate(self, patternInfo):
		"""Aligns the robot with the patttern in front of it, determines where it is using that pattern,
		then aligns itself with the path it should take to the next node. Returns True if the robot has 
		arrived at it's destination, otherwise, False."""
		# will stop the while loop running while the fixed action finishes
		pattern, center, relArea, angleToPattern = patternInfo
		location, patternOrientation, targetRelativeArea = OlinGraph.patternLocations.get(pattern, (None, None, 0))
		
		if location == None:
			self.stopColorSearching()
			return False

		self.pathTraveled.append(location)
		print "Path travelled so far:\n", self.pathTraveled
		self.fixedActs.align(targetRelativeArea)
				
		if location == self.destination:
			print "Arrived at destination."
			return True

		path = OlinGraph.olin.getShortestPath(location, self.destination)
		self.fixedActs.turnToNextTarget(path, patternOrientation)
		self.stopColorSearching()
		return False


	def sideSweep(self):
		"""Turns the robot left and right in order to check for patterns at its sides."""
		stepsFor180Degrees = 82

		self.fixedActs.turnByAngle(-90)
		with self.robot.moveControl.lock:
			self.robot.moveControl.paused = True
		try:
			twist = Twist()
			twist.linear.x = 0.0
			twist.angular.z = -0.4
			r = rospy.Rate(10)
			for i in xrange(stepsFor180Degrees):
				self.robot.moveControl.move_pub.publish(twist)
				r.sleep()
				pattern = self.mcs.getHorzPatterns()
				# Will ensure that the robot is not at too acute an angle with the pattern
				if pattern != None and abs(pattern[3]) < 45:
#					return self.locate(pattern)
					return True
		except CvBridgeError as e:
			print e
		finally:
			twist = Twist() # default to no motion
			self.robot.moveControl.move_pub.publish(twist)
			with self.robot.moveControl.lock:
				self.robot.moveControl.paused = False # start up the continuous motion loop again	
		self.fixedActs.turnByAngle(-90)
		return False

	

	def startColorSearching(self):
		"""Turn on MCS functions"""
		self.colorSearching = True
		self.colorInvestigator.enable()

	def stopColorSearching(self):
		"""Turn off MCS functions"""
		self.colorSearching = False
		self.colorInvestigator.disable()
	
	def startObstacleForce(self):
		""""""
		for behave in self.obsForceList:
			behave.enable()

	def stopObstacleForce(self):
		""""""
		for behave in self.obsForceList:
			behave.disable()


if __name__=="__main__":
  rospy.init_node('Planner')
  Planner().run(5000)

