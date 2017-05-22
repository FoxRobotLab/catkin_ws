""" ========================================================================
 * FeatureType.py
 *
 *  Created on: May 2016
 *  Author: susan
 *
 * This is a parent class for the different image features. Really doesn't do
 * much except allow some sharing of code. Should be treated as an abstract
 * class, make child classes, don't use it directly.
 *
========================================================================="""

import math
import cv2
import numpy
#import FoxQueue
#import OutputLogger


class FeatureType:
    """Abstract class, parent of individual features like Hough lines or color signature."""

    def __init__(self, image, logger, thresh=0.0, maxVal=0.0):
        """Takes in an image and the logger and sets up instance variables"""
        self.logger = logger
        self.image = image
        self.goodThresh = thresh
        self.maxValue = maxVal

    def displayFeaturePics(self, windowName, startX, startY):
        """Given a window name and a starting location on the screen, this creates
        an image that represents the color signature and displays it."""
        # Override this in the child class
        displayPic = self.image.copy()
        cv2.imshow(windowName, displayPic)
        cv2.moveWindow(windowName, startX, startY)

    def evaluateSimilarity(self, otherHL):
        """Given another HoughLines object, evaluate the similarity.
        Starts by making a priority queue for each line here.
        It then adds each line in the otherLines to queues for the lines that match.
        It then uses a greedy algorithm to pick matches (not optimal, but efficient).
        """
        # Override this in the child class
        # return self.maxValue

    def _normalizeSimValue(self, simValue):
        """Takes in a similarity value and normalizes it to be between 0 and 100. However, it divides the
        range of possible values (given by self.maxValue, into two parts, the "good" side, values less than
        self.goodThresh, and the "bad" side, values greater than self.goodThresh. This way, it stretches the "good"
        side to be values between 0 and 50, and the bad side to be values between 50 and 100."""
        if simValue < self.goodThresh:
            ratio = simValue / float(self.goodThresh)
            normedVal = ratio * 50.0
        else:
            rangeSize = self.maxValue - self.goodThresh
            distFromBottom = simValue - self.goodThresh
            ratio = distFromBottom / rangeSize
            normedVal = 50.0 + (ratio * 50.0)
        return normedVal



