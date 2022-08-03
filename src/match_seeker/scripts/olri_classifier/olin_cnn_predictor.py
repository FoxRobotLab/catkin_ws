#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
olin_cnn_predictor.py
Authors: Avik Bosshardt, Angel Sylvester and Maddie AlQatami
Based on olin_test.py, written by JJ Lim Summer 2018
Date: Summer 2019

Has some methods for predicting cells/headings based on robot image input.
Requires some tweaks depending on whether it's used with matchPlanner.py or
independently. ~~See comments below~~

Most important is to make sure that image cleaning/preprocessing is done
correctly to match the data the the CNN used to predict was trained on.
Info about this can be found in Summer 2019 journal table of experimental
results.

Note: no longer compatible with Summer 2018 data!

Summer 2020 START HERE and in Localizer.py to further incorporate new CNNs
into robot behavior.
--------------------------------------------------------------------------------"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import cv2
import time
import numpy as np
import rospy
import random
import sys
import os
from tensorflow import keras
from paths import DATA, frames, checkPts

# sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors

from turtleControl import TurtleBot
from cnn_cell_predictor_2019 import CellPredictor2019

# from OlinWorldMap import WorldMap

# uncomment to use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

def precision(y_true, y_pred):
    """Precision metric.

    Use precision in place of accuracy to evaluate models that have multiple outputs. Otherwise it's relatively
    unhelpful. The values returned during training do not represent the accuracy of the model. Use get_accuracy
    after training to evaluate models with multiple outputs.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision

def get_prediction(self, image, mapGraph,odomLoc):
    potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    # cv2.imshow('turtle view', image)
    # cv2.waitKey(10)
    # for testing


    try:
        self.lastCell = int(mapGraph.convertLocToCell(odomLoc))
    except TypeError:
        self.lastCell = mapGraph.findClosestNode(odomLoc)[0]

    cleaned_for_headings = self.clean_image(image, data='cell_channel')
    head_pred = self.heading_model.predict(cleaned_for_headings)
    best_head = potentialHeadings[np.argmax(head_pred[0])]
    self.lastHeading = np.argmax(head_pred[0])



    ### Uncomment for use with matchPlanner.py to run robot ###
    # Heading inputs are on a 0-7 scale, NOT 0-360
    cleaned_for_cells = self.clean_image(image,data='heading_channel')
    pred = self.cell_model.predict(cleaned_for_cells)
    #print(pred)
    sorted_pred = np.argsort(pred)[0][-3:][::-1]  # top 3 best matches, with best match at index 0
    #print (sorted_pred)

    best_cells_xy = []
    for i in sorted_pred:
        pred_cell = int(i)
        predXY = mapGraph.getLocation(pred_cell)
        pred_xyh = (predXY[0], predXY[1], best_head)
        best_cells_xy.append(pred_xyh)

    best_scores = []
    for i in sorted_pred:
        best_scores.append((pred[0][i])*100)

    for i in range(1,3):
        if best_scores[i] < 20:
            best_cells_xy[i] = best_cells_xy[0]
            best_scores[i] = best_scores[0]
    cell = mapGraph.convertLocToCell(best_cells_xy[0])

    return best_scores, best_cells_xy


if __name__ == "__main__":
    recent_n_matches = []
    robot = TurtleBot()
    worldMap = WorldMap()
    robot.pauseMovement()
    lastCell = -1
    lastHeading = -1

    cellPredictor = CellPredictor2019(loaded_checkpoint = "cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
                                      testData=DATA)

    lastpreds = []
    predValues = []
    while (not rospy.is_shutdown()):
        turtle_image, _ = robot.getImage()

        prediction = get_prediction(turtle_image, worldMap)
        lastpreds.append(prediction)
        if len(lastpreds) > 10:
            lastpreds.pop(0)

        predValues.append(prediction[0])

        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        if (ch == "q"):
            break
        time.sleep(0.1)

