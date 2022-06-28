#!/usr/bin/env python2.7

import rospy
import cv2
import sys
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image

def callbackTopCell(data):
        global topCell
        topCell = data.data

def callbackTopCellProb(data):
        global topCellProb
        topCellProb = data.data

def callbackTopHeading(data):
        global topHeading
        topHeading = data.data

def callbackTopHeadingProb(data):
        global topHeadingProb
        topHeadingProb = data.data

def image_callback(data):
        global image_array
        image_array = data

def listener():
    global image_sub
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("TopCell", String, callbackTopCell)
    rospy.Subscriber("TopHeading", String, callbackTopHeading)
    rospy.Subscriber("TopCell Prob", String, callbackTopCellProb)
    rospy.Subscriber("TopHeading Prob", String, callbackTopHeadingProb)
    image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, image_callback)

def getImage(x = 0, y = 0, width = 640, height = 480):
    """Method typically called by other threads, gets the next image available. If no images are available yet,
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

if __name__ == "__main__":

    # robot = TurtleBot()
    # robot.pauseMovement()
    image_sub, image_array = None, None
    topCell, topCellProb, topHeading, topHeadingProb = None, None, None, None
    listener()

    while (not rospy.is_shutdown()):
        turtle_image = getImage()
        if turtle_image and topCell and topCellProb and topHeading and topHeadingProb:
            rgbCell_text = "RGB Cell: " + topCell + " " + "{:.3f}".format(
                topCellProb)
            rgbHeading_text = "RGB Heading: " + topHeading + " " + "{:.3f}".format(
                topHeadingProb)

            turtle_image_text = cv2.putText(img=turtle_image, text=rgbCell_text, org=(50, 50), fontScale=1,
                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 255, 255))
            turtle_image_text = cv2.putText(img=turtle_image_text, text=rgbHeading_text, org=(50, 100), fontScale=1,
                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 255, 255))
            cv2.imshow("Turtle Image", turtle_image_text)

        turtle_image, topCell, topCellProb, topHeading, topHeadingProb = None, None, None, None, None
        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        if (ch == "q"):
            break

    cv2.destroyAllWindows()
    image_sub.unregister()
    # robot.exit()

    #rospy.spin()
