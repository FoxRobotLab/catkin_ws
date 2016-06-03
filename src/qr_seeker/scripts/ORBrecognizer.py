""" ========================================================================
 * ORBrecognizer.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 *  The ORBrecognizer object computes the similarity of objects using the ORB 
 *  feature detector. Is its own class purely so that it can use the methods 
 *  implemented in FeatureType.
 *
========================================================================="""


import math
import random
import time
import cv2
import numpy as np
import partitionAlg
import FeatureType
import FoxQueue
import OutputLogger

class ORBrecognizer(FeatureType.FeatureType):
    """Holds data about ORB keypoints found in the input picture."""
    def __init__(self, image, logger):
        """Takes in an image and a logger and builds keypoints and descriptions. 
        Also initializes other instance variables that are needed"""
	FeatureType.FeatureType.__init__(self, image, logger, 50.0, 100.0)
        self.image = image
        self.orb = cv2.ORB_create()
        self.kp, self.des = self.orb.detectAndCompute(self.image, None)
        self.matches = None
	self.goodMatches = None
        



    def evaluateSimilarity(self, otherFeature):
    	"""Given two images along with their features, calculates their similarity."""

    	if self.des is None and otherFeature.des is None:
	    return self._normalizeSimValue(0)
	elif self.des is None or otherFeature.des is None:
	    return  self._normalizeSimValue(100)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors.
        self.matches = bf.match(self.des,otherFeature.des)
	
	sortedMatches = sorted(self.matches, key = lambda x: x.distance)
	self.goodMatches = [mat for mat in self.matches if mat.distance < 25]
	#print "Good match number:", len(self.goodMatches)
	
	#matchImage = cv2.drawMatches(self.image, self.kp, otherFeature.image, otherFeature.kp, self.goodMatches, 
	                             #None, matchColor = (255, 255, 0), singlePointColor=(0, 0, 255))
	#cv2.imshow("Match Image", matchImage)
	#cv2.waitKey(50)
	matchNum = min(100, len(self.goodMatches))
	return self._normalizeSimValue(100 - matchNum)		



       
    def displayFeaturePics(self, windowName, startX, startY):
        """Given a window name and a starting location on the screen, it completely
        ignores them and proceeds to print the image representing the matches as it desires."""
        outIm = self.image.copy()
        img2 = cv2.drawKeypoints(self.image,self.kp, outIm, color=(0,255,0), flags=0)

        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, startX, startY)
        cv2.imshow(windowName, img2)
    