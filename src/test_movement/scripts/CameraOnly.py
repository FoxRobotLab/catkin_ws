#!/usr/bin/env python

#from roslib import message

import cv2
import rospy
import threading

from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy

from sensor_msgs.msg import Image



class ImageSensor():
  def __init__(self):
    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("/camera/rgb/image_color",Image,self.image_callback)
    self.image_sub = rospy.Subscriber("/camera/image_decompressed",Image,self.image_callback)
    cv2.namedWindow("name", 1)

  def image_callback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError, e:
      print e
    cv2.imshow("name", cv_image)
    cv2.waitKey(1)

  def exit(self):
    self.image_sub.unregister()

  
class TurtleBot:

  def __init__(self):
    self.imageControl = ImageSensor()

 
  def exit(self):
    print("************** RECEIVED ON SHUTDOWN **************")
    self.imageControl.exit()

if __name__=="__main__":

  rospy.init_node('CameraOnly')
  controller = TurtleBot()
  rospy.on_shutdown(controller.exit)
  rospy.spin()


