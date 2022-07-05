#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
collectData.py
Authors: Bea Bautista, Yifan Wu

This program reads frames from the robot camera and saves and timestamps selected frames while saving all frames to videos

To run this file:
1.Cutie must be launched (minimal, astra, and teleop)
2. In terminal run command: rosrun match_seeker scripts/buildingDatabases/collectData.py

To end this program (if it does not end as expected),
processes must be killed in the terminal using kill -9 <Python2.7 PID>

Each time this code is run, frames are saved inside match_seeker/res
--------------------------------------------------------------------------------"""
import cv2
import rospy
import sys
import os
#import datetime
import time
#sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image

#from OlinWorldMap import WorldMap

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

def saveToFolder(img, folderName, frameNum):
    fName = nextFilename(frameNum)
    pathAndName = folderName + fName
    try:
        cv2.imwrite(pathAndName, img)
    except:
        print("Error writing file", frameNum, pathAndName)


def nextFilename(num):
    fTempl = "frames{0:04d}.jpg"
    fileName = fTempl.format(num)
    return fileName

def saveVideo(destDir, hgt=480, wid=640):
    rospy.init_node('datacollector', anonymous=True, disable_signals = True)
    image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, image_callback)
    #timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    timestamp = "{}".format(time.strftime("%m%d-%H:%M"))
    videoName = destDir + '/' + timestamp + ".avi"
    frameFolder = destDir + '/' + timestamp + 'frames/'
    os.mkdir(frameFolder)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(videoName, fourcc, 25.0, (wid, hgt), 1)
    frameNum = 0
    counter = 0
    while not rospy.is_shutdown():
        frame = getImage()
        vidWriter.write(frame)
        cv2.imshow("frames", frame)
        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        counter += 1
        if counter == 10:
            saveToFolder(frame, frameFolder, frameNum)
            frameNum += 1
            counter = 0
        if ch == "q":
            cv2.destroyAllWindows()
            vidWriter.release()
            image_sub.unregister()
            print("Robot shutdown start")
            rospy.signal_shutdown("End")
            print("Robot shutdown complete")

if __name__ == "__main__":
    try:
        image_array = None
        saveVideo('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/classifier2022Data')
    except rospy.ROSInterruptException:
        pass
