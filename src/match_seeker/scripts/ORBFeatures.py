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
        self.goodMatches = [mat for mat in sortedMatches if mat.distance < 35] #orig 25
        # self.altGood1 = [mat for mat in sortedMatches if mat.distance < 50]
        # self.altGood2 = sortedMatches[0 : int(0.3*len(sortedMatches))]
        # scoreGood2 = [v.distance for v in self.altGood2]
        # if len(scoreGood2) > 0:
        #     meanGood2 = sum(scoreGood2) / len(scoreGood2)
        # else:
        #     meanGood2 = 0.0
        # print "Good match number:", len(self.goodMatches)

        betterMatches = []
        removed = []
        for m in self.goodMatches:
            pt1 = self.kp[m.queryIdx].pt
            pt2 = otherFeature.kp[m.trainIdx].pt
            end1 = tuple(np.round(pt1).astype(int))
            end2 = tuple(np.round(pt2).astype(int))
            if abs(end2[1] - end1[1]) < 80:
                betterMatches.append(m)
            else:
                removed.append(m)

        matchNum = min(200, len(betterMatches))
        normedMatchNum = (200-matchNum)/2.0 #self._normalizeSimValue(200 - matchNum)
        # if verbose:
        #     img, offset = self.joinImages(self.image,otherFeature.image)
            # matchImage = self.drawMatches(img, offset, self.kp, otherFeature.kp, betterMatches, removed,
            #                               matchColor=(255, 255, 0), remColor = (0,0,255))  # , singlePointColor=(0, 0, 255))
            # matchImage = self.drawMatches(matchImage,self.kp, otherFeature.image, otherFeature.kp,removed, matchColor=(0,0,255))
            # cv2.imshow("Match Image", matchImage)
            # cv2.waitKey(20)
            # self.logger.log("---------------------- evaluateSimilarity --------------------")
            # self.logger.log("My descriptors: " + str(len(self.des)))
            # self.logger.log("Other descrips: " + str(len(otherFeature.des)))
            # self.logger.log("Good matches:   " + str(len(betterMatches)))
            # # self.logger.log("Alt Good 1:     " + str(len(self.altGood1)))
            # self.logger.log("All matches:    " + str(len(self.matches)))
            # self.logger.log("Score:      " + str(normedMatchNum))
            # self.logger.log("----------------------        DONE        --------------------")
        return normedMatchNum


    def displayFeaturePics(self, windowName, startX, startY):
        """Given a window name and a starting location on the screen, it completely
        ignores them and proceeds to print the image representing the matches as it desires."""
        outIm = self.image.copy()
        img2 = cv2.drawKeypoints(self.image, self.kp, outIm, color=(0, 255, 0), flags=0)

        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, startX, startY)
        cv2.imshow(windowName, img2)

    def joinImages(self,img1,img2):
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

        offset = np.array([width1, 0])

        return new_img, offset


    def drawMatches(self, img, offset, kp1, kp2, matches, removes, matchColor=None, remColor = None):
        """Draws lines between matching keypoints of two images, previously placed side by side.
        Keypoints not in a matching pair are not drawn.
        draws circles around each keypoint, with line segments connecting matching pairs.
        You can tweak the r, thickness, and figsize values as needed.
        Args:
            img: An openCV image ndarray in a grayscale or color format.
            kp1: A list of cv2.KeyPoint objects for img1.
            kp2: A list of cv2.KeyPoint objects for img2.
            matches: A list of DMatch objects whose trainIdx attribute refers to
            img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
            removes: the same for matches that were reomved for being too far apart in y-direction.
            color, remColor: The color of the circles and connecting lines drawn on the images.
            A 3-tuple for color images, a scalar for grayscale images.  If None, these
            values are randomly generated.
        """


        # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
        r = 5
        thickness = 1
        if matchColor and remColor:
            c = matchColor
            n = remColor
        for m in matches:
            # Generate random color for RGB/BGR and grayscale images as needed.
            if not matchColor:
                c = np.random.randint(0, 256, 3) if len(img.shape) == 3 else np.random.randint(0, 256)

            # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
            # wants locs as a tuple of ints.
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            end1 = tuple(np.round(pt1).astype(int))
            end2 = tuple(np.round(pt2).astype(int) + offset)
            cv2.line(img, end1, end2, c, thickness)
            cv2.circle(img, end1, r, c, thickness)
            cv2.circle(img, end2, r, c, thickness)

        for rem in removes:
            if not remColor:
                n = np.random.randint(0, 256, 3) if len(img.shape) == 3 else np.random.randint(0, 256)
            pt1 = kp1[rem.queryIdx].pt
            pt2 = kp2[rem.trainIdx].pt

            end1 = tuple(np.round(pt1).astype(int))
            end2 = tuple(np.round(pt2).astype(int) + offset)
            cv2.line(img, end1, end2, n, thickness)
            cv2.circle(img, end1, r, n, thickness)
            cv2.circle(img, end2, r, n, thickness)

        return img
