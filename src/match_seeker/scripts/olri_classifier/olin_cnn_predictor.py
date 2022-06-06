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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import time
import numpy as np
import rospy
import random
import sys
import os
from tensorflow import keras
frames
sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from turtleControl import TurtleBot
from OlinWorldMap import WorldMap

# uncomment to use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class OlinTest(object):
    def __init__(self):
        self.recent_n_matches = []
        # rospy.init_node("olin_test") #uncomment to run independent of match_planner
        self.robot = TurtleBot()
        self.robot.pauseMovement()


        self.lastCell = 0
        self.lastHeading = 0



        ### Load model and weights from the specified checkpoint
        self.mean = np.load('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/DATA/TRAININGDATA_100_500_mean.npy')
        # self.model = keras.models.load_model('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/0725181447_olin-CPDr-CPDr-CPDr-DDDDr-L_lr0.001-bs128-weighted/00-745-0.71.hdf5')

        self.heading_model = keras.models.load_model("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/heading_acc9517_cellInput_250epochs_95k_NEW.hdf5",
                                                     compile = True)
        self.cell_model = keras.models.load_model("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/CHECKPOINTS/cell_acc9705_headingInput_155epochs_95k_NEW.hdf5",
                                                     compile = True)
        self.recent_n_max= 5

    def precision(self,y_true, y_pred):
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

    def clean_image(self, image, data = 'old'):
        image_size = 100
        if data == 'old': #compatible with olin_cnn 2018
            resized_image = cv2.resize(image, (image_size, image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            depth = 1
        elif data == 'vgg16': #compatible with vgg16 network for headings
            image = cv2.resize(image, (170, 128))
            x = random.randrange(0, 70)
            y = random.randrange(0, 28)
            image = image[y:y + 100, x:x + 100]
            depth = 3
        elif data == 'cell_channel':
            resized_image = cv2.resize(image, (image_size, image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            cell_arr = self.lastCell * np.ones((image_size, image_size, 1))
            image = np.concatenate((np.expand_dims(image,axis=-1),cell_arr),axis=-1)
            depth = 2
        elif data == 'heading_channel':
            resized_image = cv2.resize(image, (image_size, image_size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            cell_arr = self.lastHeading * np.ones((image_size, image_size, 1))
            image = np.concatenate((np.expand_dims(image,axis=-1),cell_arr),axis=-1)
            depth = 2
        else: #compatible with olin_cnn 2019
            image = cv2.resize(image, (170, 128))
            x = random.randrange(0, 70)
            y = random.randrange(0, 28)
            cropped_image = image[y:y + 100, x:x + 100]
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            depth = 1
        cleaned_image = np.array([image], dtype="float") \
            .reshape(1, image_size, image_size, depth)
        return cleaned_image
self.cell_model
    def test_turtlebot(self):

        lastpreds = []
        predValues = []
        while (not rospy.is_shutdown()):
            turtle_image, _ = self.robot.getImage()

            if len(lastpreds) <= 10:
                lastpreds.append(self.get_prediction(turtle_image,WorldMap()))
            else:
                lastpreds.pop(0)
                lastpreds.append(self.get_prediction(turtle_image,WorldMap()))

            #print(potentialHeadings[max(set(lastpreds), key=lastpreds.count)])
            predValues.append(self.get_prediction(turtle_image,WorldMap())[0])

            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if (ch == "q"):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

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
    test = OlinTest()
    test.test_turtlebot()
    # print(test.get_prediction(cv2.imread('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/moreframes/frame0000.jpg')))
