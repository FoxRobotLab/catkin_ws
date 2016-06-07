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
import ORBrecognizer
import OlinGraph
import numpy as np



class UpdateCamera( threading.Thread ):

    def __init__(self, bot):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.runFlag = True
        self.robot = bot
        self.frameAverageStallThreshold = 20
        self.frame = None
        self.stalled = False
        self.qrInfo = None
        self.orbInfo = None

    def orbScan(self, image):
        orbScanner = ORBrecognizer.ORBrecognizer()
        result = orbScanner.scanImages(image)
        #if result is none then orbScanner did not find enough points
        self.orbInfo =  result


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
        if result is None:
            print "Scan failed"
        else:
            for symbol in img:
                pass
            del(img)
            result = symbol.data.decode(u'utf-8')
            print "Data found:", result
            self.qrInfo = result

    def run(self):
        time.sleep(.5)
        runFlag = True
        cv2.namedWindow("TurtleCam", 1)
        timesImageServed = 1
        while(runFlag):
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
            # if orb is not None:
            #     if qrCode is not None:
            #         """pass QR info to locate"""
            #     else:
            #         """Access Fixed FixedActions"""

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

    def getImageData(self):
        with self.lock:
            orbInfo = self.orbInfo
            qrInfo = self.qrInfo
        return orbInfo, qrInfo

    def getImageDims(self):
        with self.lock:
            w, h, _ = self.frame.shape
        return w, h

class qrPlanner(object):

    def __init__(self):
        self.robot = turtleQR.TurtleBot()
        self.brain = self.setupPot()
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        #self.image, times = self.robot.getImage()
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

                    if iterationCount > 250:
                        #print ("STEPPING THE BRAIN")
                        self.brain.step()
                    else:
                        time.sleep(.01)

                    # if sweepTime < 5000:
                    #     sweepTime += 1
                    # else:
                    #     if self.sideSweep():
                    #         break
                    #     sweepTime = 0

                    print "Got to check self.ImageMatching"
                    if self.imageMatching:
                        print "YESS self.ImageMatching"
                        orbInfo, qrInfo = self.camera.getImageData()
                        if orbInfo != None:
                            sweepTime = 0
                            if self.locate(orbInfo, qrInfo):
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
        currBrain = PotentialFieldBrain.PotentialFieldBrain(self.robot)
        return currBrain


    def locate(self, orbInfo, qrInfo):
        """Aligns the robot with the orbInfo in front of it, determines where it is using that orbInfo
        by seeing the QR code below. Then aligns itself with the path it should take to the next node.
        Returns True if the robot has arrived at it's destination, otherwise, False."""
        location, codeOrientation, targetRelativeArea = OlinGraph.codeLocations.get(qrInfo, (None, None, 0))
        print "REACHED LOCATE"
        if qrInfo is not None:
            """Read things"""
            if location is None:
                """If we are lost and can not be found"""
                self.stopImageMatching()
                # return False

        # orbInfo = self.fixedActs.align(targetRelativeArea, self)
        if orbInfo == None:
            return False

        momentInfo = self.findORBContours(orbInfo)

        self.fixedActs.align(momentInfo)

        self.pathTraveled.append(location)
        print ("Path travelled so far:\n", self.pathTraveled)

        if location == self.destination:
            print ("Arrived at destination.")
            return True
        print ("Finished align, now starting turning to next target")

        self.fixedActs.turnToNextTarget(location, self.destination, codeOrientation)
        self.stopImageMatching()
        return False


    """Big idea: there may be multiple contours of blue areas in the image, so we need to
    find contours of the good keypoints, taking into account that some of these may be noise."""
    def findORBContours(self, goodKeyPoints):
        w, h = self.camera.getImageDims()
        black = np.zeros((w, h), dtype='uint8')

        #ALL the keypoints, the list of matched keypoints within some range
        keypoints, goodMatches = goodKeyPoints

        # For each pair of points we have between both images
        for mat in goodMatches:

            # Get the matching keypoints for each of the images from the match struct DMatch
            img_idx = mat.queryIdx
            (x, y) = keypoints[img_idx].pt

            #draw large-ish white circles around each of the keypoints
            cv2.circle(black, (int(x), int(y)), 10, (255, 255, 255), -1)

        #use closing to eliminate the white spots not in "clusters" - the noise keypoints
        kernel = np.ones((15,15),np.uint8)
        closing = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        #find contour of the remaining white blob. There may be small noise blobs, but we're
        #pretty sure that the largest is the one we want.
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.contourArea(contours[0])

        maxArea = 0
        largestContour = 0

        for i in range (0, len(contours)):
            contour = contours[i]
            area = cv2.contourArea(contour)
            if area > maxArea:
                maxArea = area
                largestContour = contour

        M = cv2.moments(largestContour)
        imageArea = cv2.contourArea(largestContour)
        relativeArea = imageArea / float(w*h)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # TODO: Dissappear that string
        return ("", (cx,cy), relativeArea)

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
        for t in xrange(5):
            imageInfo = self.ORBrecog.scanImages()
            if imageInfo != None:
                return self.locate(imageInfo)
            time.sleep(0.2)

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
                # imageMatch = self.ORBrecog.scanImages()
                # Will ensure that the robot is not at too acute an angle with the code? Necessary?
                imageMatch = None
                if imageMatch != None:
                    return self.locate(imageMatch)
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
