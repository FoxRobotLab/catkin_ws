#!/usr/bin/env python

""" ========================================================================
qrPlanner.py
Created: June, 2016
Starts a thread to grab images from the camera and scan then for ORB features
and QR codes. Imports zbar to read QR codes that are in the turtlebots view.
======================================================================== """

import cv2
from datetime import datetime
from collections import deque
import rospy
import time
import threading
import zbar
from PIL import Image
import ORBrecognizer
import string



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
        self.orbScanner = ORBrecognizer.ORBrecognizer()
        self.qrScanner = zbar.ImageScanner()

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

            if image is not None:
                self.qrScan(image)
                self.orbScan(image)

            keypress = chr(cv2.waitKey(50) & 255)

            if keypress == 't':
                cv2.imwrite("/home/macalester/catkin_ws/src/speedy_nav/res/captures/cap-"
                                + str(datetime.now()) + ".jpg", image)
                print "Image saved!"
            if keypress == 'q':
                break

            with self.lock:
                runFlag = self.runFlag

    def orbScan(self, image):
        result = self.orbScanner.scanImage(image)
        #if result is none then orbScanner did not find enough points
        with self.lock:
            self.orbInfo = result

    def qrScan(self, image):
        self.qrScanner.parse_config('enable')
        bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("bwimg", bwImg)
        #cv2.waitKey(0)
        pil_im = Image.fromarray(bwImg)
        pic2 = pil_im.convert("L")
        wid, hgt = pic2.size
        #print("wid, hgt,", wid, hgt)
        raw = pic2.tobytes()

        img = zbar.Image(wid, hgt, 'Y800', raw)
        result = self.qrScanner.scan(img)
        #print "RESULT", result
        if result == 0:
            #print "Scan failed"
            pass
        else:
            #print ("img is ", img)
            for symbol in img:
                #print "symbol did indeed get assigned"
                pass
            del(img)
            codeData = symbol.data.decode(u'utf-8')
            #print "Data found:", codeData
            list = string.split(codeData)
            #print(list)
            nodeNum = list[0]
            nodeCoord = list[1] + ' ' + list[2]
            nodeName = ''
            for i in range(3, len(list)):
                nodeName = nodeName + ' ' + list[i]

            #nodeNum, nodeCoord, nodeName = string.split(codeData)
            with self.lock:
                self.qrInfo = (int(nodeNum), nodeCoord, nodeName)

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
