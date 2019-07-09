#!/usr/bin/env python2.7
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
import olri_classifier.olin_factory as factory #use for matchPlanner.py
# import olin_factory as factory #use for olin_cnn.py
from datetime import datetime
sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts')
from turtleControl import TurtleBot
# uncomment to use CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''


class OlinTest(object):
    def __init__(self, recent_n_max):
        self.recent_n_cells = []
        cellF = open(factory.paths.maptocell_path, 'r')
        cellDict = dict()
        for line in cellF:
            if line[0] == '#' or line.isspace():
                continue
            parts = line.split()
            cellNum = parts[0]
            locList = [int(v) for v in parts[1:]]
            cellDict[cellNum] = locList
        self.cell_data =  cellDict
        #rospy.init_node("olin_test") #uncomment to run independent of match_planner
        self.robot = TurtleBot()
        self.robot.pauseMovement()


        cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path, allow_pickle=True).item()
        self.intlabel_to_cell_dict = dict()
        for cell in cell_to_intlabel_dict.keys():
            self.intlabel_to_cell_dict[cell_to_intlabel_dict[cell]] = cell

        self.paths = factory.paths
        self.hyperparameters = factory.hyperparameters
        self.image = factory.image
        self.cell = factory.cell

        self.mean = np.load(factory.paths.train_mean_path)
        ### Load model and weights from the specified checkpoint
        self.model = keras.models.load_model(factory.paths.checkpoint_name)
        self.model.load_weights(factory.paths.checkpoint_name)
        self.heading_model = keras.models.load_model("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/VGG16-CellsHeadings-Dropout-10epochs/NEWTRAININGDATA_100-05-0.13.hdf5", custom_objects = {'precision':self.precision})
        print("*** Model restored: ", factory.paths.checkpoint_name)

        self.recent_n_max=recent_n_max

    def precision(self,y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = keras.backend.sum(keras.backend.round(keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.backend.epsilon())
        return precision

    def clean_image(self, image, data = 'old'):
        if data == 'old': #compatible with olin_cnn 2018
            resized_image = cv2.resize(image, (factory.image.size, factory.image.size))
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            depth = 1
        elif data == 'vgg16': #compatible with vgg16 network for headings
            image = cv2.resize(image, (170, 128))
            x = random.randrange(0, 70)
            y = random.randrange(0, 28)
            image = image[y:y + 100, x:x + 100]
            depth = 3
        else: #compatible with olin_cnn 2019
            image = cv2.resize(image, (170, 128))
            x = random.randrange(0, 70)
            y = random.randrange(0, 28)
            cropped_image = image[y:y + 100, x:x + 100]
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            image = np.subtract(gray_image, self.mean)
            depth = 1
        cleaned_image = np.array([image], dtype="float") \
            .reshape(1, factory.image.size, factory.image.size, depth)
        return cleaned_image

    def test_turtlebot(self):
        # cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path).item()
        # intlabel_to_cell_dict = dict()
        # for cell in cell_to_intlabel_dict.keys():
        #     intlabel_to_cell_dict[cell_to_intlabel_dict[cell]] = cell
        # print(intlabel_to_cell_dict)
        lastpreds = []
        predValues = []
        while (not rospy.is_shutdown()):
            turtle_image, _ = self.robot.getImage()

            if len(lastpreds) <= 10:
                lastpreds.append(self.get_prediction(turtle_image))
            else:
                lastpreds.pop(0)
                lastpreds.append(self.get_prediction(turtle_image))

            #print(potentialHeadings[max(set(lastpreds), key=lastpreds.count)])
            predValues.append(self.get_prediction(turtle_image)[0])
            print (self.get_prediction(turtle_image))

            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if (ch == "q"):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

    def get_prediction(self, image, mapGraph):
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        cleaned_image = self.clean_image(image)
        pred = self.model.predict(cleaned_image)
        cleaned_for_headings = self.clean_image(image, data='vgg16')
        head_pred = self.heading_model.predict(cleaned_for_headings)

        sorted_pred = np.argsort(pred)[0][-3:][::-1] #top 3 best matches, with best match at index 0
        best_head = potentialHeadings[np.argmax(head_pred[0][-8:])]


        best_cells_xy = []
        for i in sorted_pred:
            pred_cell = self.intlabel_to_cell_dict[i]
            predXY = mapGraph.getLocation(pred_cell)
            pred_xyh = (predXY[0], predXY[1], best_head)
            best_cells_xy.append(pred_xyh)

        best_scores = []
        for i in sorted_pred:
            best_scores.append((pred[0][i])*100)

        for i in range(1,3):
            if best_scores[i] < 33:
                best_cells_xy[i] = best_cells_xy[0]
                best_scores[i] = best_scores[0]


        return best_scores, best_cells_xy

if __name__ == "__main__":
    test = OlinTest(5)
    # test.test_turtlebot()
    # print(test.get_prediction(cv2.imread('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/moreframes/frame0000.jpg')))
