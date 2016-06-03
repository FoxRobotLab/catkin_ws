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
		time.sleep(.5)
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

			code = chr(cv2.waitKey(50) & 255)

			if code == 't':
				cv2.imwrite("/home/macalester/catkin_ws/src/speedy_nav/res/captures/cap-" 
					+ time.strftime("%b%d%a-%H%M%S.jpg"), image)
				print "Image saved!"
			if code == 'q':
				break

			with self.lock:
				runFlag = self.runFlag

	def isStalled(self):
		"""Returns the status of the camera stream"""
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

		image, times = self.robot.getImage()		
		self.mcs = MCS.MultiCamShift(image.shape)
		self.colorSearching = True

		self.camera = UpdateCamera(self.robot, self.mcs)

		"""self.destination = 999999999
		while self.destination > len(OlinGraph.nodeLocs):
			self.destination = int(input("Enter destination index: "))

		self.fixedActs = FixedActions.FixedActions(self.robot, self.mcs, self.camera)


		self.colorInv = FieldBehaviors.ColorInvestigate(self.mcs)
		#self.brain.add( self.colorInv )	
		self.brain.add( FieldBehaviors.KeepMoving() )
		# self.brain.add( FieldBehaviors.RandomWander(1000) )
		for i in range(10, 100, 10):
			self.brain.add( FieldBehaviors.ObstacleForce(i/100.0, 50) )

		self.pathTraveled = []"""



	def run(self,runtime = 120):
		"""Runs the pattern location program for the duration of 'runtime'"""
		timeout = time.time()+runtime
		self.camera.start()
		timeToWaitAfterStall = 30
		# iterators
		iterationCount = 0
		ignoreColorTime = 0
		sweepTime = 0
		sinceLastStall = 0
                print "Planner.run starting while loop"
		while time.time() < timeout and not rospy.is_shutdown():	

			if not self.camera.isStalled():
                                print " -------------------------------- camera not stalled"
				sinceLastStall += 1
				if 30 < sinceLastStall:
					"""bumper = self.robot.getBumperStatus()
					if bumper != 0:
						while not self.bumperReact(bumper):
							cv2.waitKey(300)
							bumper = self.robot.getBumperStatus()
						self.stopColorSearching()
					print " ---   inside if 30 < sinceLastStall""""
					iterationCount += 1
#					print "======================================", '\titer: ', iterationCount 
					"""if iterationCount > 250: 
                                                print "STEPPING THE BRAIN"
						self.brain.step()
					else:
						time.sleep(.01)

					if sweepTime < 5000:
						sweepTime += 1
					else:
						if self.sideSweep():
							break
						sweepTime = 0"""
	
					if self.colorSearching:
						pattern = self.mcs.getHorzPatterns()
						# Prevents the robot from coming at the pattern from too sharp an angle
						if pattern != None and abs(pattern[3]) < 45:
							sweepTime = 0
							if self.locate(pattern[0]):
								break
					else:
						if ignoreColorTime < 1000:
							ignoreColorTime += 1
						else:
							self.startColorSearching()
							ignoreColorTime = 0
			else:
				sinceLastStall = 0
		self.camera.haltRun()
		self.camera.join()
		self.brain.stopAll()
	

					
	def setupPot(self):
		"""Helpful function takes optional robot code (the six-digit Fluke board number). If code
		is given, then this connects to the robot. Otherwise, it connects to a simulated robot, and
		then creates a SubsumptionBrain object and returns it."""
		currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
		return currBrain


	def locate(self, pattern):
		"""Aligns the robot with the patttern in front of it, determines where it is using that pattern,
		then aligns itself with the path it should take to the next node. Returns True if the robot has 
		arrived at it's destination, otherwise, False."""
		location, patternOrientation, targetRelativeArea = OlinGraph.patternLocations.get(pattern, (None, None, 0))
		if location == None:
			self.stopColorSearching()
			return False

		pattern = self.fixedActs.align(targetRelativeArea, self)
		if pattern == None:
			return False

		self.pathTraveled.append(location)
		print "Path travelled so far:\n", self.pathTraveled
				
		if location == self.destination:
			print "Arrived at destination."
			return True
		print "Finished align, now starting turning to next target"

		self.fixedActs.turnToNextTarget(location, self.destination, patternOrientation)
		self.stopColorSearching()
		return False


	def bumperReact(self, bumper_state):
		"After the bumper is triggered, responds accordingly and return False if robot should stop"""
		if bumper_state == 0:
			return True

		
		#These codes correspond to the wheels dropping
		if bumper_state == 16 or bumper_state == 28:
			return False

		self.robot.backward(0.2, 3)
		cv2.waitKey(300)

		patternInfo = None
		for t in xrange(5):
			patternInfo = self.mcs.getHorzPatterns()
			if patternInfo != None:
				return self.locate(patternInfo[0])
			time.sleep(0.2)

		#left side of the bumper was hit
		if bumper_state == 2:
			self.fixedActs.turnByAngle(45)
		#right side of the bumper was hit
		if bumper_state == 1 or bumper_state == 3:
			self.fixedActs.turnByAngle(-45)
		return True



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
					return self.locate(pattern[0])
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
		self.colorInv.enable()

	def stopColorSearching(self):
		"""Turn off MCS functions"""
		self.colorSearching = False
		self.colorInv.disable()

	def exit(self):
		self.camera.haltRun()
		self.mcs.exit()
	
if __name__=="__main__":
	rospy.init_node('Planner')
	plan = Planner()
	plan.run(5000)
	rospy.on_shutdown(plan.exit)
	rospy.spin()

