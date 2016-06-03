#!/usr/bin/env python

import TurtleBot
import cv2
from datetime import datetime
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
import rospy
import time

class ImageCapture(object):

	def __init__(self):
        self.robot = TurtleBot.TurtleBot()
		image, times = self.robot.getImage()		

		self.camera = UpdateCamera(self.robot)

		self.lock = threading.Lock()
		self.runFlag = True
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
			
			cv2.imshow("TurtleCam", image)

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

	def exit(self):
		with self.lock:
			self.runFlag = False
	
if __name__=="__main__":
	rospy.init_node('imageCapture')
	camera = ImageCapture()
	camera.run()
	rospy.on_shutdown(camera.exit)
	rospy.spin()

