#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
cnn_predictor_2022.py
Authors: Bea Bautista, Shosuke Noma, Yifan Wu
Based on olin_cnn_predictor.py

Has some methods for predicting cells/headings based on robot image input.
Requires some tweaks depending on whether it's used with matchPlanner.py or
independently. ~~See comments below~~

--------------------------------------------------------------------------------"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import cv2
import time
# import rospkg
# import imp
# foo = imp.load_source('rospy','/opt/ros/melodic/lib/python2.7/dist-packages/rospy/__init__.py')
# foo.MyClass()
# foo = imp.load_source('rospy','/opt/ros/melodic/lib/python2.7/dist-packages/rospy/__init__.pyc')
# foo.MyClass()
import rospy
# import roslibpy
import numpy as np
import random
import sys
import os
from tensorflow import keras
#sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from paths import DATA, checkPts
from turtleControl import TurtleBot
from cnn_cell_predictor_2019 import CellPredictor2019
from cnn_cell_predictor_RGBinput import CellPredictorRGB
from cnn_heading_predictor_2019 import HeadingPredictor
from cnn_heading_predictor_RGBinput import HeadingPredictorRGB
from std_msgs.msg import String

#from OlinWorldMap import WorldMap

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == "__main__":
    robot = TurtleBot()
    robot.pauseMovement()
    pubTopCell = rospy.Publisher('Top Cell', String, queue_size=10)
    pubTopCellProb = rospy.Publisher('Top Cell Prob', String, queue_size=10)
    pubTopHeading = rospy.Publisher('Top Heading', String, queue_size=10)
    pubTopHeadingProb = rospy.Publisher('Top Heading Prob', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    # cellPredictor = CellPredictor2019(loaded_checkpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
    #                                   testData = DATA, checkPtsDirectory = checkPts)
    #
    # headingPredictor = HeadingPredictor(loaded_checkpoint="heading_acc9517_cellInput_250epochs_95k_NEW.hdf5",
    #                                     testData = DATA, checkPtsDirectory = checkPts)

    cellPredictorRGB = CellPredictorRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="latestCellPredictorFromPrecision5810.hdf5"
    )
    cellPredictorRGB.buildNetwork()

    headingPredictorRGB = HeadingPredictorRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="latestHeadingPredictorFromPrecision5810.hdf5",
    )
    headingPredictorRGB.buildNetwork()

    headingnumber = input("Initial heading: ")
    cell = input("Initial cell: ")
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    while (not rospy.is_shutdown()):
        turtle_image, _ = robot.getImage()

        # 2022 RGB Cell Predictor Model
        pred_cellRGB, output_cellRGB = cellPredictorRGB.predictSingleImageAllData(turtle_image)
        topThreePercs_cellRGB, topThreeCells_cellRGB = cellPredictorRGB.findTopX(3, output_cellRGB)
        print("cell prediction RGB model:", pred_cellRGB)
        print("Top three:", topThreeCells_cellRGB, topThreePercs_cellRGB)

        # 2022 RGB Heading Predictor Model
        pred_headingRGB, output_headingRGB = headingPredictorRGB.predictSingleImageAllData(turtle_image)
        topThreePercs_headingRGB, topThreeHeadingID_headingRGB = cellPredictorRGB.findTopX(3, output_headingRGB)
        print("heading prediction RGB model:", potentialHeadings[pred_headingRGB])
        print("Top three:", topThreeHeadingID_headingRGB, topThreePercs_headingRGB)

        # # 2019 Cell Predictor Model
        # pred_cell, output_cell = cellPredictor.predictSingleImageAllData(turtle_image, headingnumber)
        # topThreePercs_cell, topThreeCells_cell = cellPredictorRGB.findTopX(3, output_cell)
        # print("cell prediction 2019 model:", pred_cell)
        # print("Top three:", topThreeCells_cell, topThreePercs_cell)
        # cell = pred_cell
        #
        # # 2019 Heading Predictor Model
        # pred_heading, output_heading = headingPredictor.predictSingleImageAllData(turtle_image, cell)
        # topThreePercs_heading, topThreeHeadingID_heading = cellPredictorRGB.findTopX(3, output_heading)
        # print("heading prediction 2019 model:", potentialHeadings[pred_heading])
        # print("Top three:", topThreeHeadingID_heading, topThreePercs_heading)
        # headingnumber = potentialHeadings[pred_heading]

        topCell = str(topThreeCells_cellRGB[0])
        topCellProb = "{:.3f}".format(topThreePercs_headingRGB[0])
        topHeading = str(potentialHeadings[topThreeHeadingID_headingRGB[0]])
        topHeadingProb = "{:.3f}".format(topThreePercs_headingRGB[0])
        pubTopCell.publish(topCell)
        pubTopCellProb.publish(topCellProb)
        pubTopHeading.publish(topHeading)
        pubTopHeadingProb.publish(topHeadingProb)

        rgbCell_text = "RGB Cell: " + topCell + " " + "{:.3f}".format(
            topCellProb) + "%"
        rgbHeading_text = "RGB Heading: " + topHeading + " " + "{:.3f}".format(
            topHeadingProb) + "%"
        turtle_image_text = cv2.putText(img = turtle_image, text = rgbCell_text, org = (50,50), fontScale= 1, fontFace= cv2.FONT_HERSHEY_DUPLEX, color = (0,255,255))
        turtle_image_text = cv2.putText(img = turtle_image_text, text = rgbHeading_text, org = (50,100), fontScale= 1, fontFace= cv2.FONT_HERSHEY_DUPLEX, color = (0,255,255))

        cv2.imshow("Turtle Image", turtle_image_text)

        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        if (ch == "q"):
            break
        if (ch=="o"):
            headingnumber = input("new heading: ")
            cell = input("new cell: ")

        #rospy.loginfo(topCell)

        time.sleep(0.1)

    cv2.destroyAllWindows()
    robot.exit()

