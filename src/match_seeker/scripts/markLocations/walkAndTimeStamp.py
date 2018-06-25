import os
import time
import turtleControl

import cv2
import numpy as np
import locsForFrames

# import readMap
import src.match_seeker.scripts.markLocations.readMap as readMap



#image, _ = self.robot.getImage()
#self.robot.stop()

class LabeledFramesByMeter(object):
    def __init__(self, outputFilePath, mapFile, robot):

        self.outputFilePath = outputFilePath
        self.mapFile = mapFile
        self.robot = robot

        self.currLoc = (0, 0)
        self.currHeading = 0
        self.imgFileList = []

        # instance variables to hold displayed images
        self.mainImg = None
        self.origMap = None
        self.currMap = None
        self.currFrame = None

        # Instance variables to hold outcome data
        self.labeling = dict()



    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("Main")
        cv2.namedWindow("Image")
        cv2.namedWindow("Map")
        cv2.moveWindow("Main", 30, 50)
        cv2.moveWindow("Image", 30, 550)
        cv2.moveWindow("Map", 700, 50)


    def _convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return (mapXMeters, mapYMeters)


    def _convertWorldToMap(self, worldX, worldY):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        pixelX = worldX * 20.0
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.mapWid - 1 - pixelY
        mapY = self.mapHgt - 1 - pixelX
        return (int(mapX), int(mapY))





if __name__ == "main":
    catkinPath = "/Users/johnpellegrini/"
    basePath= "PycharmProjects/catkin_ws/src/match_seeker/"

    robot = turtleControl.TurtleBot()


    liveAssigningLocations = LabeledFramesByMeter(mapFile=catkinPath + basePath + "res/map/olinNewMap.txt",
                                                  outputFilePath= "/Users/johnpellegrini/Desktop/recordedVideos/",
                                                  robot = robot)
    liveAssigningLocations._setupWindows()

