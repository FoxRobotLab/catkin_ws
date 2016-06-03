""" ========================================================================
 * turtleQR.py
 *
 *  Created on: June 2016
 *
 *  The turtleQR.py file borrows code from TurtleBot.py in speedy_nav.
 * Four classes: one for the TurtleBot object, and then three thread classes -
 * MovementControlThread, ImageSensorThread, and SensorThread.  TurtleBot
 * initialises all the threads and provides movement/utility methods -
 * turning, bumper status, find angle to wall, and so on.
 * Uses Utils. Uses CvBridge to be able to use images from kinect with OpenCV/ROS
 *
========================================================================="""

#!/usr/bin/env python

#from roslib import message
import rospy
import threading
import sys

#from math import *

from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy

from Utils import *
from sensor_msgs.msg import Image

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from create_node.msg import TurtlebotSensorState


class MovementControlThread( threading.Thread ):

	def __init__(self):
		threading.Thread.__init__(self)
		self.lock = threading.Lock()
		self.runFlag = True
		self.translate = 0.0
		self.rotate = 0.0
		self.rateLimiter = rospy.Rate(10) #10Hz
		self.paused = False

		self.move_pub = rospy.Publisher('mobile_base/commands/velocity', Twist)

	def run(self):
		twist = Twist()
		with self.lock:
			twist.linear.x = self.translate
			twist.angular.z = self.rotate
			self.runFlag = True
		runFlag = True
		paused = False
		try:
			while(runFlag):
				with self.lock:
					paused = self.paused
				if not paused:
					self.move_pub.publish(twist)
					self.rateLimiter.sleep()
					with self.lock:
						twist.linear.x = self.translate
						twist.angular.z = self.rotate
						runFlag = self.runFlag
		except:
			print e
		finally:
			twist = Twist() # default to no motion
			self.move_pub.publish(twist)

	def setRotate(self, amount):
		with self.lock:
			self.rotate = amount

	def setTranslate(self, amount):
		with self.lock:
			self.translate = amount

	def setMovement(self, translate, rotate):
		with self.lock:
			self.translate = translate
			self.rotate = rotate

	def stopMovement(self):
		with self.lock:
			self.translate = 0.0
			self.rotate = 0.0

	def timedMovement(self, translate, rotate, seconds):
		with self.lock:
			self.paused = True
		try:
			twist = Twist()
			twist.linear.x = translate
			twist.angular.z = rotate
			r = rospy.Rate(10) #10Hz
			for i in range (int(seconds*10)):
				self.move_pub.publish(twist)
				r.sleep()
		except:
			print e
		finally:
			twist = Twist() # default to no motion
			self.move_pub.publish(twist)
			with self.lock:
				self.paused = False # start up the continuous motion loop again


	def haltRun(self):
		with self.lock:
			self.runFlag = False
			self.translate = 0.0
			self.rotate = 0.0


class ImageSensorThread( threading.Thread ):
	def __init__(self):
		threading.Thread.__init__(self)
		self.lock = threading.Lock()
		self.bridge = CvBridge()

		self.imageProcessed = False
		self.image_array = None
		self.savedImage = None
		#self.image_sub = rospy.Subscriber("/camera/rgb/image_color",Image,self.image_callback)
		self.image_sub = rospy.Subscriber("/camera/image_decompressed", Image, self.image_callback)

		self.runFlag = True

	def run(self):
		with self.lock:
			self.runFlag = True
		runFlag = True
		while runFlag:
			rospy.sleep(0.2)
			with self.lock:
				runFlag = self.runFlag
		self.image_sub.unregister()


	def image_callback(self, data):
		with self.lock:
			self.image_array = data
			self.imageProcessed = False

	def getImage(self, x, y, width, height):
		with self.lock:
			data = self.image_array
			processed = self.imageProcessed
		if processed == False:
			try:
				if data == None:
					sys.stdout.write("Waiting for camera data")
					while data == None:
						sys.stdout.write(".")
						sys.stdout.flush()
						rospy.sleep(0.2)
						if rospy.is_shutdown():
							return
						with self.lock:
							data = self.image_array
					print " Done!"
				cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
			except CvBridgeError, e:
				print e

			retval = cv_image[y:y+height, x:x+width]
			with self.lock:
				self.savedImage = retval
				self.imageProcessed = True
			self.numTimesImgServed = 1
		else:
			with self.lock:
				retval = self.savedImage
			self.numTimesImgServed += 1
		return retval , self.numTimesImgServed


	def exit(self):
		print "image sensor shutdown received"
		with self.lock:
			self.runFlag = False

class SensorThread( threading.Thread ):
	def __init__(self):
		print "sensor thread init"
		threading.Thread.__init__(self)
		self.lock = threading.Lock()
		self.bridge = CvBridge()

		self.depth_array = None
		self.depth_sub = rospy.Subscriber("/camera/depth_decompressed", Image, self.depth_callback)
		
		self.sensor_state = None
		self.sensor_sub = rospy.Subscriber("/mobile_base/sensors/core", TurtlebotSensorState, self.sensor_callback)

		self.runFlag = True

	def run(self):
		with self.lock:
			self.runFlag = True
		runFlag = True
			
		while runFlag:
			rospy.sleep(0.1)
			with self.lock:
				runFlag = self.runFlag
		self.sensor_sub.unregister()
		self.depth_sub.unregister()


	def sensor_callback(self, data):
		with self.lock:
			self.sensor_state = data
	

	def depth_callback(self, data):
		with self.lock:
			self.depth_array = data


	def getDims(self):
		with self.lock:
			data = self.depth_array
		if data == None:
			return 0,0
		return data.width, data.height

	def getSensorState(self):
		with self.lock:
			state = self.sensor_state
		return state


	def getDepth(self, x, y, width, height):
		with self.lock:
			data = self.depth_array
		try:
			if data == None:
				sys.stdout.write("Waiting for depth data")
				while data == None:
					sys.stdout.write(".")
					sys.stdout.flush()
					if rospy.is_shutdown():
						return
					rospy.sleep(0.1)
					with self.lock:
						data = self.depth_array
				print " Done!"
			cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
		except CvBridgeError as e:
			print e
		numpy_array = numpy.asarray(cv_image)	 
		retval = numpy_array[y:y+height, x:x+width]
		return retval

	def exit(self):
		print "depth sensor shutdown received"
		with self.lock:
			self.runFlag = False



class TurtleBot(object):

	def __init__(self):

		#self.moveControl = MovementControlThread()
		#self.moveControl.start()
		
		#self.sensorControl = SensorThread()
		#self.sensorControl.start()

		self.imageControl = ImageSensorThread()
		self.imageControl.start()

		#rospy.on_shutdown(self.exit)
	def findAngleToWall(self):
		width, height = self.sensorControl.getDims()
		sectionWidth = width / 30.0
		sectionHeight = height / 5.0
		# box = c,r,w,h
		leftBox = int(sectionWidth * 10), int(sectionHeight * 2), int(sectionWidth), int(sectionHeight)
		rightBox = int(sectionWidth * 19), int(sectionHeight * 2), int(sectionWidth), int(sectionHeight)
		
		leftDist = numpy.mean(self.sensorControl.getDepth(*leftBox))
		rightDist = numpy.mean(self.sensorControl.getDepth(*rightBox))
		return getMedianAngle(leftDist, rightDist, angleBetween = 11.0)


	def forward(self, amount, seconds=None):
		if seconds == None:
			self.moveControl.setMovement(amount, 0.0)
		else:
			self.moveControl.timedMovement(amount, 0.0, seconds)

	def backward(self, amount, seconds):
		if seconds == None:
			self.moveControl.setMovement(-amount, 0.0)
		else:
			self.moveControl.timedMovement(-amount, 0.0, seconds)

	def turnLeft(self, amount, seconds):
		if seconds == None:
			self.moveControl.setMovement(0.0, amount)
		else:
			self.moveControl.timedMovement(0.0, amount, seconds)

	def turnRight(self, amount, seconds):
		if seconds == None:
			self.moveControl.setMovement(0.0, amount)
		else:
			self.moveControl.timedMovement(0.0, -amount, seconds)

	def stop(self):
		self.moveControl.stopMovement()

	def translate(self, amount):
		self.moveControl.setTranslate(amount)

	def rotate(self, amount):
		self.moveControl.setRotate(amount)

	def move(self, translate, rotate):
		self.moveControl.setMovement(translate, rotate)

	def getDepth(self, x = 0, y = 0, width = 640, height = 480):
		return self.sensorControl.getDepth(x, y, width, height)

	def getImage(self, x = 0, y = 0, width = 640, height = 480):
		return self.imageControl.getImage(x, y, width, height)

	def getBumperStatus(self):
		state = self.sensorControl.getSensorState()
		if state == None:
			return 0
		return state.bumps_wheeldrops

	# units are either "deg" or "s2"
	def turnTo(self, angle, units):
		pass

	def turnBy(self, angle, units):
		pass

	def exit(self):
		print("************** RECEIVED ON SHUTDOWN **************")
		#self.moveControl.haltRun()
		#self.moveControl.join()
		#self.sensorControl.exit()
		#self.sensorControl.join()
		self.imageControl.exit()
		self.imageControl.join()

if __name__=="__main__":

	rospy.init_node('test_movement')
	controller = TurtleBot()
	rospy.on_shutdown(controller.exit)
	rospy.spin()
