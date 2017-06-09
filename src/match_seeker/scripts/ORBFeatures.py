""" ========================================================================
 * ORBFeatures.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 * Updated Summer 2017 and integrated into qr_seeker project
 * Author: Susan Fox, Kayla Beckham, Xinyu Yang
 *
 *  The ORBFeatures object computes the similarity of objects using the ORB
 *  feature detector. Is its own class purely so that it can use the methods
 *  implemented in FeatureType.
 *
========================================================================="""

import math
import random
import time
import cv2
import numpy as np
#import partitionAlg
import FeatureType
#import FoxQueue
#import OutputLogger


class ORBFeatures(FeatureType.FeatureType):
    """Holds data about ORB keypoints found in the input picture."""

    def __init__(self, image, logger, orbFinder):
        """Takes in an image and a logger and an ORB finding object and builds keypoints and descriptions.
        Also initializes other instance variables that are needed"""
        FeatureType.FeatureType.__init__(self, image, logger, 50.0, 100.0)
        self.image = image
        self.orb = orbFinder
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        self.kp, self.des = self.orb.detectAndCompute(self.image, None)
        self.matches = None
        self.goodMatches = None
        self.maxValue = 100


    def evaluateSimilarity(self, otherFeature, verbose = False):
        """Given two images along with their features, calculates their similarity."""
        if self.des is None and otherFeature.des is None:
            return self._normalizeSimValue(0)
        elif self.des is None or otherFeature.des is None:
            return self._normalizeSimValue(self.maxValue)

        # Match descriptors.
        self.matches = self.matcher.match(self.des, otherFeature.des)
        sortedMatches = sorted(self.matches, key=lambda x: x.distance)
        self.goodMatches = [mat for mat in sortedMatches if mat.distance < 25]

        # print "Good match number:", len(self.goodMatches)
        # matchImage = self.drawMatches(self.image, self.kp, otherFeature.image, otherFeature.kp, self.goodMatches,
        #                               matchColor = (255, 255, 0)) #, singlePointColor=(0, 0, 255))
        # cv2.imshow("Match Image", matchImage)
        # cv2.waitKey(0)
        matchNum = min(100, len(self.goodMatches))
        normedMatchNum = self._normalizeSimValue(100 - matchNum)
        if verbose:
            print "---------------------- evaluateSimilarity --------------------"
            print "My descriptors: ", len(self.des)
            print "Other descrips: ", len(otherFeature.des)
            print "Good matches:   ", len(self.goodMatches)
            print "All matches:    ", len(self.matches)
            print "Match number:   ", matchNum
            print "Raw score:      ", 100 - matchNum
            print "Normed score:   ", normedMatchNum
            print "----------------------        DONE        --------------------"
        return normedMatchNum


    def displayFeaturePics(self, windowName, startX, startY):
        """Given a window name and a starting location on the screen, it completely
        ignores them and proceeds to print the image representing the matches as it desires."""
        outIm = self.image.copy()
        img2 = cv2.drawKeypoints(self.image, self.kp, outIm, color=(0, 255, 0), flags=0)

        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, startX, startY)
        cv2.imshow(windowName, img2)


    def drawMatches(self, img1, kp1, img2, kp2, matches, matchColor=None):
        """Draws lines between matching keypoints of two images.
        Keypoints not in a matching pair are not drawn.
        Places the images side by side in a new image and draws circles
        around each keypoint, with line segments connecting matching pairs.
        You can tweak the r, thickness, and figsize values as needed.
        Args:
            img1: An openCV image ndarray in a grayscale or color format.
            kp1: A list of cv2.KeyPoint objects for img1.
            img2: An openCV image ndarray of the same format and with the same
            element type as img1.
            kp2: A list of cv2.KeyPoint objects for img2.
            matches: A list of DMatch objects whose trainIdx attribute refers to
            img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
            color: The color of the circles and connecting lines drawn on the images.
            A 3-tuple for color images, a scalar for grayscale images.  If None, these
            values are randomly generated.
        """
        # We're drawing them side by side.  Get dimensions accordingly.
        # Handle both color and grayscale images.
        height1 = img1.shape[0]
        height2 = img2.shape[0]
        width1 = img1.shape[1]
        width2 = img2.shape[1]
        newHeight = max(height1, height2)
        newWidth = width1 + width2
        if len(img1.shape) == 3:
            new_shape = (newHeight, newWidth, img1.shape[2])
        elif len(img1.shape) == 2:
            new_shape = (newHeight, newWidth)
        new_img = np.zeros(new_shape, type(img1.flat[0]))
        # Place images onto the new image.
        new_img[0:height1, 0:width1] = img1
        new_img[0:height2, width1:newWidth] = img2

        # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
        r = 5
        thickness = 1
        if matchColor:
            c = matchColor
        for m in matches:
            # Generate random color for RGB/BGR and grayscale images as needed.
            if not matchColor:
                c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
            # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
            # wants locs as a tuple of ints.
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            offset = np.array([width1, 0])
            end1 = tuple(np.round(pt1).astype(int))
            end2 = tuple(np.round(pt2).astype(int) + offset)
            cv2.line(new_img, end1, end2, c, thickness)
            cv2.circle(new_img, end1, r, c, thickness)
            cv2.circle(new_img, end2, r, c, thickness)
        return new_img
