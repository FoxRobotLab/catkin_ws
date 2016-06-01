#!/usr/bin/env python

#from roslib import message
import rospy
import cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#import sensor_msgs.point_cloud2 as pc2
#from sensor_msgs.msg import PointCloud2, PointField

#from geometry_msgs.msg import Twist


class image_converter:

  def __init__(self):
    cv.NamedWindow("Image window", 1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/depth/image",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv(data, "passthrough")
    except CvBridgeError, e:
      print e

    cv.ShowImage("Image window", cv_image)
    cv.WaitKey(3)


class depth_info:

  def __init__(self):
    self.scan_sum = rospy.Subscriber("/camera/depth/points", PointCloud2, self.callback)

  def callback(self, data):
    
    #find middle point of data
    mid_x = int(data.width/2)
    mid_y = int(data.height/2)
    mid_point_data = pc2.read_points(data, uvs=[[mid_x, mid_y]])
    mid_point_depth = next(mid_point_data)
    print mid_point_depth


if __name__=="__main__":

  ic = image_converter()
# di = depth_info()
  rospy.init_node('test_movement')
  #pub = rospy.Publisher('test_movement/cmd_vel', Twist)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print "Shutting down"
  cv.DestroyAllWindows()


  '''
  try:
    r = rospy.Rate(10) #10Hz
    while not rospy.is_shutdown():
      print "hello"
      twist = Twist()
      twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
      twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0.5
      pub.publish(twist)
      r.sleep()
  except:
    print e

  finally:
    twist = Twist()
    twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
    pub.publish(twist)
  '''
