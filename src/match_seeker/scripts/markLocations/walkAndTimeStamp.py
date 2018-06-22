import os
import time
import rospy

import cv2
import numpy as np
import locsForFrames

# import readMap
import src.match_seeker.scripts.markLocations.readMap as readMap

class LabeledFramesByMeter(object):
    def __init__(self, outputFilePath, mapFile):

        self.outputFilePath = outputFilePath
        self.mapFile = mapFile




if __name__ == "main":
    catkinPath = "/Users/johnpellegrini/"
    basePath= "PycharmProjects/catkin_ws/src/match_seeker/"

    robot = turtleControl.TurtleBot()


    liveAssigningLocations = LabeledFramesByMeter(mapFile=catkinPath + basePath + "res/map/olinNewMap.txt",
                                                  outputFilePath= "/Users/johnpellegrini/Desktop/recordedVideos/")

