#!/usr/bin/env python

import FieldBehaviors
import PotentialFieldBrain
import FixedActions
import TurtleBot
import MultiCamShift as MCS
import cv2
from datetime import datetime
import OlinGraph

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

	def run(self):
		runFlag = True
		cv2.namedWindow("TurtleCam 9000", 1)
		while(runFlag):
			image, timesImageServed = self.robot.getImage()
			with self.lock:
				if timesImageServed > 30:
					if self.stalled == False:
						print "Camera Stalled!"
					self.stalled = True
				else:
					self.stalled = False
					
				
			frame = self.mcs.update(image.copy())
			cv2.imshow("TurtleCam 9000", frame)

			code = chr(cv2.waitKey(10) & 255)

			if code == 't':
				cv2.imwrite("/home/macalester/catkin_ws/src/speedy_nav/res/captures/cap-" + str(datetime.now()) + ".jpg", image)
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
		self.destination = 999999999
		while self.destination > len(OlinGraph.nodeLocs):
			self.destination = int(input("Enter destination index: "))

		self.fixedActs = FixedActions.FixedActions(self.robot, self.mcs)

		self.camera = UpdateCamera(self.robot, self.mcs)

		self.colorInv = FieldBehaviors.ColorInvestigate(self.mcs)
		#self.brain.add( self.colorInv )	
		self.brain.add( FieldBehaviors.KeepMoving() )
		# self.brain.add( FieldBehaviors.RandomWander(1000) )
		for i in range(10, 100, 10):
			self.brain.add( FieldBehaviors.ObstacleForce(i/100.0, 20) )



	def run(self,runtime = 120):
		"""Runs the pattern location program for the duration of 'runtime'"""
		timeout = time.time()+runtime
		self.camera.start()

		# iterators
		iterationCount = 0
		ignoreColorTime = 0
		sweepTime = 0

		while time.time() < timeout and not rospy.is_shutdown():
			#rospy.sleep(0.1)


			if not self.camera.isStalled():
				iterationCount += 1
#				print "======================================", '\titer: ', iterationCount  
				self.brain.step()

				if sweepTime < 5000:
					sweepTime += 1
				else:
					self.sideSweep()
					sweepTime = 0


			if self.colorSearching:
				pattern = self.mcs.getHorzPatterns()
				if pattern != None and abs(pattern[3]) < 30:
					sweepTime = 0
					if self.locate(pattern):
						print "Arrived at destination."
						break
#			else:
#				if ignoreColorTime < 800:
#					ignoreColorTime += 1
#				else:
#					self.startColorSearching()
#					ignoreColorTime = 0
			
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
		"""Aligns the robot with the patttern in front of it, then determines where it is using that pattern."""
		# will stop the while loop running while the fixed action finishes
		pattern, center,calcalating angle perspective relArea, angleToPattern = patternInfo
		#self.stopColorSearching()
		location, patternOrientation = OlinGraph.patternLocations[pattern]
		actualAngle = (patternOrientation - angleToPattern) % 360
		path = OlinGraph.olin.getShortestPath(location, self.destination)
		self.fixedActs.align(path, patternOrientation)
		if location == self.destination:
			return True
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
				if pattern != None and abs(pattern[3]) < 30:
					return self.locate(pattern)
		except:
			print e
		finally:
			twist = Twist() # default to no motion
			self.robot.moveControl.move_pub.publish(twist)
			with self.robot.moveControl.lock:
				self.robot.moveControl.paused = False # start up the continuous motion loop again	
		self.fixedActs.turnByAngle(-90)
		return False

#		self.fixedActs.turnByAngle(-90)
#		for x in xrange(18):
#			self.fixedActs.turnByAngle(10)
#			cv2.waitKey(300)
#			pattern = self.mcs.getHorzPatterns()
#			if pattern != None:
#				return self.locate(pattern)
#		self.fixedActs.turnByAngle(-90)
#		return False

	

	def startColorSearching(self):
		"""Turn on MCS functions"""
		self.colorSearching = True
		self.colorInv.enable()

	def stopColorSearching(self):
		"""Turn off MCS functions"""
		self.colorSearching = False
		self.colorInv.disable()
	
if __name__=="__main__":
  rospy.init_node('Planner')
  Planner().run(5000)

