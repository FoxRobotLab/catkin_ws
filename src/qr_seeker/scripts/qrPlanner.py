#!/usr/bin/env python

""" ========================================================================
qrPlanner.py
Created: June, 2016
This file borrows code from the Planner.py in Speedy_nav. This file
uses FixedActions.py and PotentialFieldBrain.py. This file imports zbar to
read QR codes that are in the turtlebots view.
======================================================================== """

import turtleQR
import cv2
from datetime import datetime
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
import rospy
import time
import threading
from geometry_msgs.msg import Twist
import zbar
from PIL import Image
import FixedActions
import PotentialFieldBrain
from ORBrecognizer import *
import OlinGraph



class UpdateCamera( threading.Thread ):

	def __init__(self, bot):
		threading.Thread.__init__(self)
		self.lock = threading.Lock()
		self.runFlag = True
		self.robot = bot
		self.frameAverageStallThreshold = 20
		self.frame = None
        self.stalled = False

	def orbScan(self, image):
		orbScanner = ORBrecognizer()
		result = orbScanner.scanImages(image)
		#if result is true it found the thing, else not

	def scanImage(self, image):   #TODO: Move this below with turning on QR search?
		scanner = zbar.ImageScanner()
		print(scanner)
		scanner.parse_config('enable')
		bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		pil_im = Image.fromarray(bwImg)
		pic2 = pil_im.convert("L")
		wid, hgt = pic2.size
		raw = pic2.tobytes()

		img = zbar.Image(wid, hgt, 'Y800', raw)
		result = scanner.scan(img)

		if result == 0:
			print "Scan failed"
		else:
			for symbol in img:
				pass
			del(img)
			data = symbol.data.decode(u'utf-8')
			print "Data found:", data

	def run(self):
		time.sleep(.5)
		runFlag = True
		cv2.namedWindow("TurtleCam", 1)
		timesImageServed = 1
		while(runFlag):
elf.robot.getImage().sh
			image, timesImageServed = self.robot.getImage()
			self.frame = image

			with self.lock:
				if timesImageServed > 20:
					if self.stalled == False:
						print "Camera Stalled!"
					self.stalled = True
				else:
					self.stalled = False

			cv2.imshow("TurtleCam", image)
			#self.scanImage(image)
			self.orbScan(image)

			code = chr(cv2.waitKey(50) & 255)

			if code == 't':
				cv2.imwrite("/home/macalester/catkin_ws/src/speedy_nav/res/captures/cap-"
					+ str(datetime.now()) + ".jpg", image)
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

class qrPlanner(object):

	def __init__(self):
		self.robot = turtleQR.TurtleBot()
		self.brain = self.setupPot()
		self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
		image, times = self.robot.getImage()
		self.imageMatching = True

		self.camera = UpdateCamera(self.robot)

		self.destination = 999999999
		while self.destination > len(OlinGraph.nodeLocs):
			self.destination = int(input("Enter destination index: "))

		self.fixedActs = FixedActions.FixedActions(self.robot, self.camera)
		self.pathTraveled = []


	def run(self,runtime = 120):
		#Runs the program for the duration of 'runtime'"""
		timeout = time.time()+runtime
		self.camera.start()
		timeToWaitAfterStall = 30
		iterationCount = 0
		ignoreColorTime = 0
		sweepTime = 0
		sinceLastStall = 0
		#print ("Planner.run starting while loop")
		while time.time() < timeout and not rospy.is_shutdown():
			if not self.camera.isStalled():
				#print (" -------------------------------- camera not stalled")
				sinceLastStall += 1
				if 30 < sinceLastStall:
					bumper = self.robot.getBumperStatus()
					if bumper != 0:
						while not self.bumperReact(bumper):
							cv2.waitKey(300)
							bumper = self.robot.getBumperStatus()
						self.stopImageMatching()
					#print (" ---   inside if 30 < sinceLastStall")
					iterationCount += 1
					#                   print "======================================", '\titer: ', iterationCount
					if iterationCount > 250:
						#print ("STEPPING THE BRAIN")
						self.brain.step()
					else:
						time.sleep(.01)

					if sweepTime < 5000:
						sweepTime += 1
					else:
						if self.sideSweep():
							break
						sweepTime = 0

					if self.imageMatching:
						# TODO: Find image w/ ORB to locate QR code and make the below something real possibly
						# imageMatch = self.orb.returnValues()

						# TODO: have orb recognizer return the following values (around 137)?
						# cnt = contours[0]
						# M = cv2.moments(cnt)
						# imageArea = cv2.contourArea(cnt)
						# relativeArea = imageArea / float(self.fWidth * self.fHeight)
						# cx = int(M['m10'] / M['m00'])
						# cy = int(M['m01'] / M['m00'])


						# Prevents the robot from coming at the code from too sharp an angle
						if imageMatch != None and abs(imageMatch[3]) < 45:
							sweepTime = 0
							if self.locate(imageMatch[0]):
								break
					else:
						if ignoreColorTime < 1000:
							ignoreColorTime += 1
						else:
							self.startImageMatching()
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


	def locate(self, imageMatch):
		"""Aligns the robot with the imageMatch in front of it, determines where it is using that imageMatch
		by seeing the QR code below. Then aligns itself with the path it should take to the next node.
		Returns True if the robot has arrived at it's destination, otherwise, False."""
		location, codeOrientation, targetRelativeArea = OlinGraph.codeLocations.get(imageMatch, (None, None, 0))
		if location == None:
			self.stopImageMatching()
			return False

		imageMatch = self.fixedActs.align(targetRelativeArea, self)
		if imageMatch == None:
			return False

		#TODO working here
		recognizer = ORBrecognizer(imageMatch, )

		# scanner = zbar.ImageScanner() #TODO: Move here?
		# print(scanner)
		# scanner.parse_config('enable')
		# bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# pil_im = Image.fromarray(bwImg)
		# pic2 = pil_im.convert("L")
		# wid, hgt = pic2.size
		# raw = pic2.tobytes()
		#
		# img = zbar.Image(wid, hgt, 'Y800', raw)
		# result = scanner.scan(img)
		#
		# if result == 0:
		#   print
		#   "Scan failed"
		# else:
		#   for symbol in img:
		#       pass
		#   del (img)
		#   data = symbol.data.decode(u'utf-8')
		#   print("Data found:", data)

		self.pathTraveled.append(location)
		print ("Path travelled so far:\n", self.pathTraveled)

		if location == self.destination:
			print ("Arrived at destination.")
			return True
		print ("Finished align, now starting turning to next target")

		self.fixedActs.turnToNextTarget(location, self.destination, codeOrientation)
		self.stopImageMatching()
		return False


	def bumperReact(self, bumper_state):
		"After the bumper is triggered, responds accordingly and return False if robot should stop"""
		if bumper_state == 0:
			return True

		# These codes correspond to the wheels dropping
		if bumper_state == 16 or bumper_state == 28:
			return False

		self.robot.backward(0.2, 3)
		cv2.waitKey(300)

		imageInfo = None
		#for t in xrange(5):
			# TODO: Use ORB to locate image and QR code can be read below

		# left side of the bumper was hit
		if bumper_state == 2:
			self.fixedActs.turnByAngle(45)
		# right side of the bumper was hit
		if bumper_state == 1 or bumper_state == 3:
			self.fixedActs.turnByAngle(-45)
		return True


	def sideSweep(self):
		"""Turns the robot left and right in order to check for codes at its sides. Trying to minimize need for
		this because this slows down the navigation process."""
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
				# TODO: Use ORB to locate image and QR code below
				# Will ensure that the robot is not at too acute an angle with the code
				if imageMatch != None and abs(imageMatch[3]) < 45:
					return self.locate(imageMatch[0])
		except CvBridgeError as e:
			print (e)
		finally:
			twist = Twist()  # default to no motion
			self.robot.moveControl.move_pub.publish(twist)
			with self.robot.moveControl.lock:
				self.robot.moveControl.paused = False  # start up the continuous motion loop again
		self.fixedActs.turnByAngle(-90)
		return False


	def startImageMatching(self):
		"""Turn on searching functions"""
		self.imageMatching = True


	def stopImageMatching(self):
		"""Turn off searching functions"""
		self.imageMatching = False


	def exit(self):
		self.camera.haltRun()


if __name__=="__main__":
	rospy.init_node('Planner')
	plan = qrPlanner()
	plan.run(5000)
	rospy.on_shutdown(plan.exit)
	rospy.spin()