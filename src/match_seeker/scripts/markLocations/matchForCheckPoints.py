import os
import cv2
import rospy
import src.match_seeker.scripts.markLocations.readMap as readMap
import numpy as np

class Matcher(object):

    def __init__(self, walkerTextFile, autoTextFile):
        self.walkerTextFile = walkerTextFile
        self.autoTextFile = autoTextFile












if __name__ == "main":
    matchWalkerAndImages = Matcher(walkerTextFile= "",
                                   autoTextFile = "")
