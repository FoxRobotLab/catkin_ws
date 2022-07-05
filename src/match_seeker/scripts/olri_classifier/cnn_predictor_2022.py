#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
cnn_predictor_2022.py
Authors: Bea Bautista, Shosuke Noma, Yifan Wu
Based on olin_cnn_predictor.py

Code that allows for TurtleBot running on ROS Melodic to use cell and heading
predictor models to predict its current location in real time. To run the models
on Cutie, this script must be called in the terminal using:

rosrun match_seeker scripts/olri_classifier/cnn_predictor_2022.py

Cutie must be launched (minimal, astra, and teleop) for this code to run. To end this program,
processes must be killed in the terminal using kill -9 <Python2.7 PID>

Each time this code is run, logs are saved in new directories inside match_seeker/res/csvLogs2022.
These new directories include the chosen frames to be logged as well as performance metrics
of each model.
--------------------------------------------------------------------------------"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import cv2
import time
import rospy
import csv
import numpy as np
import random
import sys
import os
from tensorflow import keras
#sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from paths import DATA, checkPts, logs
from turtleControl import TurtleBot
from cnn_cell_predictor_2019 import CellPredictor2019
from cnn_cell_predictor_RGBinput import CellPredictorRGB
from cnn_heading_predictor_2019 import HeadingPredictor
from cnn_heading_predictor_RGBinput import HeadingPredictorRGB
from std_msgs.msg import String

#from OlinWorldMap import WorldMap

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def convertHeadingIndList(list):
    headings = []
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    for i in list:
        headings.append(potentialHeadings[i])
    return headings

def inTopX(item, list):
    if item in list:
        return "T"
    return "F"

if __name__ == "__main__":
    rospy.init_node('olripredictor-new', anonymous=False, disable_signals = True)
    robot = TurtleBot()
    robot.pauseMovement()

    pubTopCell = rospy.Publisher('TopCell', String, queue_size=10)
    pubTopCellProb = rospy.Publisher('TopCellProb', String, queue_size=10)
    pubTopHeading = rospy.Publisher('TopHeading', String, queue_size=10)
    pubTopHeadingProb = rospy.Publisher('TopHeadingProb', String, queue_size=10)

    cellPredictor = CellPredictor2019(loaded_checkpoint = checkPts + "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5", testData = DATA)
    headingPredictor = HeadingPredictor(loaded_checkpoint=checkPts + "heading_acc9517_cellInput_250epochs_95k_NEW.hdf5", testData = DATA)

    cellPredictorRGB = CellPredictorRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="cellPredictorRGB100epochs.hdf5"
    )
    cellPredictorRGB.buildNetwork()

    headingPredictorRGB = HeadingPredictorRGB(
        checkPointFolder=checkPts,
        loaded_checkpoint="headingPredictorRGB100epochs.hdf5"
    )
    headingPredictorRGB.buildNetwork()

    headingnumber = input("Initial heading: ")
    cell = input("Initial cell: ")
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    dirTimeStamp = "{}".format(time.strftime("%m%d%y%H%M"))
    logPath = logs + "turtleLog-" + dirTimeStamp
    photoPath = logPath + "/turtlePhotos-" + dirTimeStamp
    os.mkdir(logPath)
    os.mkdir(photoPath)
    csvLog = open(logPath + "/turtleLog-" + dirTimeStamp + ".csv", "w")
    filewriter = csv.writer(csvLog)
    filewriter.writerow(
        ["Frame", "Actual Cell", "Actual Heading",
         "Prob Actual Cell RGB", "Prob Actual Heading RGB", "Prob Actual Cell 2019", "Prob Actual Cell 2019",
         "Pred Cell 2022", "Prob Cell 2022", "Actual in Top 3", "Top 3 Cells", "Top 3 Cell Prob",
         "Pred Heading 2022", "Prob Heading 2022", "Actual in Top 3", "Top 3 Headings", "Top 3 Heading Prob",
         "Heading for 2019 Cell Pred", "Pred Cell 2019", "Prob Cell 2019", "Actual in Top 3", "Top 3 Cells", "Top 3 Cell Prob",
         "Cell for 2019 Heading Pred", "Pred Heading 2019", "Prob Heading 2019", "Actual in Top 3", "Top 3 Headings", "Top 3 Heading Prob"])

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

        # 2019 Cell Predictor Model
        pred_cell, output_cell = cellPredictor.predictSingleImageAllData(turtle_image, headingnumber)
        topThreePercs_cell, topThreeCells_cell = cellPredictorRGB.findTopX(3, output_cell)
        print("cell prediction 2019 model:", pred_cell)
        print("Top three:", topThreeCells_cell, topThreePercs_cell)

        # 2019 Heading Predictor Model
        pred_heading, output_heading = headingPredictor.predictSingleImageAllData(turtle_image, cell)
        topThreePercs_heading, topThreeHeadingID_heading = cellPredictorRGB.findTopX(3, output_heading)
        print("heading prediction 2019 model:", potentialHeadings[pred_heading])
        print("Top three:", topThreeHeadingID_heading, topThreePercs_heading)

        topCell = str(topThreeCells_cellRGB[0])
        topCellProb = "{:.3f}".format(topThreePercs_cellRGB[0])
        topHeading = str(potentialHeadings[topThreeHeadingID_headingRGB[0]])
        topHeadingProb = "{:.3f}".format(topThreePercs_headingRGB[0])
        # pubTopCell.publish(topCell)
        # pubTopCellProb.publish(topCellProb)
        # pubTopHeading.publish(topHeading)
        # pubTopHeadingProb.publish(topHeadingProb)

        rgbCell_text = "RGB Cell: " + topCell + " " + topCellProb
        rgbHeading_text = "RGB Heading: " + topHeading + " " + topHeadingProb
        turtle_image_text = cv2.putText(img = turtle_image, text = rgbCell_text, org = (50,50), fontScale= 1, fontFace= cv2.FONT_HERSHEY_DUPLEX, color = (0,255,255))
        turtle_image_text = cv2.putText(img = turtle_image_text, text = rgbHeading_text, org = (50,100), fontScale= 1, fontFace= cv2.FONT_HERSHEY_DUPLEX, color = (0,255,255))

        cv2.imshow("Turtle Image", turtle_image_text)

        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        if ch == "q":
            #break

            cv2.destroyAllWindows()
            print"Robot shutdown start"
            # robot.exit()
            rospy.signal_shutdown("End")
            print "Robot shutdown complete"
        if ch == "o":
            headingnumber = input("new heading: ")
            cell = input("new cell: ")
        if ch == "s":
            frame = "turtleIm-{}.jpg".format(time.strftime("%m%d%y%H%M"))
            cv2.imwrite(photoPath + "/" + frame, turtle_image)
            actualCell = input("Actual Cell: ")
            actualHeading = input("Actual Heading: ")
            headingTop3RGB = convertHeadingIndList(topThreeHeadingID_headingRGB)
            headingTop3 = convertHeadingIndList(topThreeHeadingID_heading)
            actualHeadingIndex = potentialHeadings.index(int(actualHeading))
            filewriter.writerow(
                [frame, actualCell, actualHeading,
                 "{:.3f}".format(output_cellRGB[int(actualCell)]), "{:.3f}".format(output_headingRGB[int(actualHeadingIndex)]),
                 "{:.3f}".format(output_cell[int(actualCell)]), "{:.3f}".format(output_heading[int(actualHeadingIndex)]),
                 topCell, topCellProb, inTopX(int(actualCell), topThreeCells_cellRGB),  str(topThreeCells_cellRGB), str(topThreePercs_cellRGB),
                 topHeading, topHeadingProb, inTopX(int(actualHeading), headingTop3RGB), str(headingTop3RGB), str(topThreePercs_headingRGB),
                headingnumber, str(topThreeCells_cell[0]), "{:.3f}".format(topThreePercs_cell[0]), inTopX(int(actualCell), topThreeCells_cell), str(topThreeCells_cell), str(topThreePercs_cell),
                cell, str(potentialHeadings[topThreeHeadingID_heading[0]]), "{:.3f}".format(topThreePercs_heading[0]), inTopX(int(actualHeading), headingTop3), str(headingTop3), str(topThreePercs_heading)])

        cell = pred_cell
        headingnumber = potentialHeadings[pred_heading]
        time.sleep(0.1)

    cv2.destroyAllWindows()
    print"out of loop"





