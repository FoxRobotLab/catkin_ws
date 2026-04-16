#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
sendFrames.py
Author: Oscar Reza B.

Reads frames from the robot camera and sends a text encoding of the frames to
port 5555 through the TCP protocol. This text is to be received by
visualizeDetector.py for object detection and display through cv2.imshow().

To run this file:
1. Cutie2 must be launched (minimal, astra, and teleop)
2. In terminal run command:
      rosrun collision_avoidance scripts/object_detection/sendFrames.py

To end this program (if it does not end as expected),
processes must be killed in the terminal using kill -9 <Python2.7 PID>

--------------------------------------------------------------------------------"""
import cv2
import rospy
import sys
import zmq
import base64
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def image_callback(data):
    global image_array
    image_array = data

def getImage(x = 0, y = 0, width = 640, height = 480):
    """Gets the next image available. If no images are available yet,
    this method blocks until one becomes available."""
    global image_array
    try:
        if image_array is None:
            sys.stdout.write("Waiting for camera image_array")
            while image_array is None:
                sys.stdout.write(".")
                sys.stdout.flush()
                rospy.sleep(0.2)
                if rospy.is_shutdown():
                    return
            print " Done!"
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image_array, "passthrough")
    except CvBridgeError, e:
        print e
    retval = cv_image[y:y + height, x:x + width]
    r, g, b = cv2.split(retval)
    retval = cv2.merge((b,g,r))

    return retval


def runVideo(hgt=480, wid=640):
  # Define counters for saving frames
  frame_count = 0
  SEND_EVERY_N_FRAMES = 10

  # Initialize ROS node
  rospy.init_node('detectorvisualizer', anonymous=True, disable_signals=True)  # disable signals such as ctrl+C
  image_sub = rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

  # Initialize ZeroMQ
  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.bind("tcp://*:5555")  # bind to port 5555

  while not rospy.is_shutdown():
    # Get frame and update count
    frame = getImage()
    frame_count += 1

    # Send frame through TCP, every N frames
    if frame_count % SEND_EVERY_N_FRAMES == 0:
      _, buffer = cv2.imencode('.jpg', frame)
      jpg_as_text = base64.b64encode(buffer)
      socket.send(jpg_as_text)

    # Display just a black rectangle
    cv2.imshow("Hit 'q' to exit", np.zeros((240, 320, 3), dtype="uint8"))
    key = cv2.waitKey(10)
    ch = chr(key & 0xFF)

    if ch == "q":
      cv2.destroyAllWindows()

      image_sub.unregister()
      print("Robot shutdown start")
      rospy.signal_shutdown("End")
      print("Robot shutdown complete")

if __name__ == "__main__":
    try:
        image_array = None
        runVideo()
    except rospy.ROSInterruptException:
        pass
