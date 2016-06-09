""" ========================================================================
 * ORBrecognizer.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 *  The ORBrecognizer object computes the similarity of objects using the ORB
 *  feature detector.
 *
========================================================================="""


import math
import random
import time
import os
import cv2
import numpy as np
import partitionAlg
import FeatureType
import FoxQueue
import OutputLogger
from operator import itemgetter
import turtleQR


class ORBrecognizer():
    """Holds data about ORB keypoints found in the input picture."""
    def __init__(self, bot):
        #self.image = None
        self.orb = cv2.ORB_create()
        #self.kp, self.des = self.orb.detectAndCompute(self.image, None)
        self.matches = None
        self.goodMatches = None
        self.robot = bot
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape


    def tryToMatchFeatures(self, orb, img1, pointInfo, img2):
        kp2, des2 = pointInfo

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)

        if des1 is None or des2 is None:
            return [], None, None
        elif len(des1) == 0 or len(des2) == 0:
            print("it thinks the key descriptions are empty")
            return [], None, None

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.match(des1,des2)

        sortedMatches = sorted(matches, key = lambda x: x.distance)
        #for mat in matches:
        #    print mat.distance
        goodMatches = [mat for mat in matches if mat.distance < 300]
        #print "Good match number:", len(goodMatches)

        matchImage = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches,
            None, matchColor = (255, 255, 0), singlePointColor=(0, 0, 255))
        cv2.imshow("Match Image", matchImage)

        return goodMatches, kp1, kp2


    def findImage(self, img, properties, itemsSought):

        colorImage = img
        answer = False

        #properties[i][2] holds the list of good points and the two sets of keypoints (for drawing)
        orb = cv2.ORB_create()
        for i in range (0, len(itemsSought)):
            goodPoints, targetKeypts, refKeypts = self.tryToMatchFeatures(orb, img, properties[i][1], properties[i][0])
            properties[i][2].append(goodPoints)
            properties[i][2].append(targetKeypts)
            properties[i][2].append(refKeypts)

            #properties[i][3] holds the len(goodPoints) - the number of good points of that item
            numGoodPoints = len(properties[i][2][0])
            properties[i][3] = numGoodPoints

        getcount = itemgetter(3)
        scores = map(getcount, properties)
        #print(scores)
        max_index, max_value = max(enumerate(scores), key=itemgetter(1))


        if max_value > 25:
            #print('The '+ str(itemsSought[max_index]) + ' sign was detected, with ' + str(max_value) + ' points')
            #retVal = (itemsSought[max_index], (cx, cy), relativeArea)
            retVal = (properties[max_index][2][1], properties[max_index][2][0]) #set of good points for the sign with the most good points
        else:
            #print('No sign was detected')
            retVal = None

        #cleanup
        for i in range (0, len(itemsSought)):
            properties[i][2] = []
            properties[i][3] = 0

        return retVal

    def colorPreprocessing(self, img, colorSample):
        img2 = img.copy()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # convert to HSV
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))   # eliminate low and high saturation and value values

        hsv_roi = cv2.cvtColor(colorSample, cv2.COLOR_BGR2HSV) # access the currently selected region and make a histogram of its hue
        #mask_roi = cv2.inRange(colorSample, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

        hist = cv2.calcHist( [hsv_roi], [0], None, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.reshape(-1)

        #print cv2.countNonZero(hist)

        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        #cv2.imshow("prob", prob)
        prob &= mask

        #opening removes the remanants in the background
        size = 15
        kernel = np.ones((size,size),np.uint8)
        prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)

        #thresholding to black and white removes the grey areas of noise left in the image from opening
        ret,prob = cv2.threshold(prob,127,255,cv2.THRESH_BINARY)

        #split into channels in order to apply the mask to each channel and merge - sure there's a better way to do this
        blue_channel, green_channel, red_channel = cv2.split(img)
        blue_channel &= prob
        green_channel &= prob
        red_channel &= prob
        cv2.merge((blue_channel, green_channel, red_channel), img)
        #cv2.imshow("prob", prob)

        #Prob at this point has black outline around the letter themselves, which is why we need
        #to do all the stuff with the contours. Blobs don't work because the white areas are too large.
        _, contours, hierarchy = cv2.findContours(prob, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.contourArea(contours[0])

        #TODO make a black mask
        rows = img.shape[0]
        cols = img.shape[1]
        black = np.zeros((rows, cols), dtype='uint8')

        for i in range (0, len(contours)):
            contour = contours[i]
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(black, [box], 0, (255,0,0), -1)

        white = cv2.countNonZero(black)

        blue_channel, green_channel, red_channel = cv2.split(img2)
        blue_channel &= black
        green_channel &= black
        red_channel &= black
        cv2.merge((blue_channel, green_channel, red_channel), img2)

        #cv2.imshow("returns", img2)
        return img2

    def initRefs(self, itemsSought):
        properties = [] #2D array used to store info on each item: item i is properties[i]

        orb = cv2.ORB_create()

        #properties[i][0] holds the reference image for the item
        for i in range (0, len(itemsSought)):
            properties.append([None, [], [], 0])

            filename = itemsSought[i] + '.jpg'
            path = "/home/macalester/catkin_ws/src/qr_seeker/res/refs/" + filename
            properties[i][0] = cv2.imread(path, 0)
            if properties[i][0] is None:
                print("Reference image", itemsSought[i], "not found")

            #properties[i][1] holds the keypoints and their descriptions for the reference image
            keypoints, descriptions = orb.detectAndCompute(properties[i][0], None)
            properties[i][1].append(keypoints)
            properties[i][1].append(descriptions)

        return properties

    #run this if you wanna test the feature recognition using a still image
    def orbScan(self, image):
        itemsSought = ['sign']
        properties = self.initRefs(itemsSought)

        filename = 'blue.jpg'
        path = "/home/macalester/catkin_ws/src/qr_seeker/res/refs/" + filename
        try:
            colorSample = cv2.imread(path)
        except:
            print "Could not read the sample color image!"

        image2 = image.copy()
        img = self.colorPreprocessing(image2, colorSample)
        return self.findImage(img, properties, itemsSought)

    def getFrameDims(self):
        """Returns the the dimmensions and depth of the camera frame"""
        return self.fWidth, self.fHeight


    def getFrameCenter(self):
        """Returns the center coordinates of the camera frame"""
        return self.fWidth / 2, self.fHeight / 2

