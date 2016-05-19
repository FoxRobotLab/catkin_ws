#!/usr/bin/env python

import rospy

import PotentialFieldBrain
import test_movement
import MultiCamShift as MCS

import cv2
import numpy
import time
from datetime import datetime

import math
import random


class KeepMoving(PotentialFieldBrain.PotentialFieldBehavior):
    """This is a brain-dead class that just reports a fixed magnitude and a heading that
    matches the robot's current heading"""

    def update(self):
        """set zero magnitude and current heading"""
        self.setVector(0.2, 0.0)

class ObjectForce(PotentialFieldBrain.PotentialFieldBehavior):

    def __init__(self, color, speedMult, camShift):
        self.color = color
        self.speedMult = speedMult
        self.multiCamShift = camShift
        self.targetSize = 0.25
            
    def update(self):
        # width = 640
        # height = 480
        # objList size will be 0 if nothing is found, 1 otherwise

        objList = self.multiCamShift.getAverageOfColor(self.color)
        if len(objList) != 0:
            ((x, y), size) = objList[0]
            angle = -((x/640.0)-0.5)*60*2
            speed = (self.targetSize-size)*3
            if angle < 0:
                angle += 360
            if speed < 0:
                angle = -angle
                if angle > 180:
                    angle -= 180
                else:
                    angle += 180
                speed = -speed/4.0
            self.setVector(speed, angle)
        else:
            self.setVector(0.0, 0.0)

class UpdateCamera(PotentialFieldBrain.PotentialFieldBehavior):

    def __init__(self, camShift):
        cv2.namedWindow("Turtlebot Camera", 1)
        self.multiCamShift = camShift
           
    def update(self):
        image = self.robot.getImage().copy()
        image2 = self.multiCamShift.update(image)
        cv2.imshow("Turtlebot Camera", image2)
        code = chr(cv2.waitKey(1) & 255)
        if code == 'c':
            cv2.imwrite("/home/macalester/catkin_ws/src/test_movement/scripts/images/captures/cap-"
                         + str(datetime.now()) + ".jpg", image2)
            print "Image saved!"

       
        self.setVector(0.0, 0.0)

# -----------------------------------------------------
# Run the demo using something like this:

def runDemo(runtime = 120):
    brain = setupPot()
    image = brain.myRobot.getImage()

    mcs = MCS.MultiCamShift(image)

    # brain.add( KeepMoving() )
    #brain.add( ObjectForce("blue", 0.2, mcs) )
    #brain.add( ObjectForce("purple", 0.2, mcs) )
    brain.add( ObjectForce("green", 0.2, mcs) )
    brain.add( UpdateCamera(mcs) )   
 
    timeout = time.time()+runtime
    while time.time() < timeout and not rospy.is_shutdown():
        #rospy.sleep(0.1)
        print("======================================")
        brain.step()

    brain.stopAll()


                    
def setupPot(robotCode = None):
    """Helpful function takes optional robot code (the six-digit Fluke board number). If code
    is given, then this connects to the robot. Otherwise, it connects to a simulated robot, and
    then creates a SubsumptionBrain object and returns it."""

    currRobot = test_movement.TurtleBot()
    currBrain = PotentialFieldBrain.PotentialFieldBrain(currRobot)
    return currBrain

if __name__=="__main__":
  rospy.init_node('TrackingField')
  runDemo(5000)

