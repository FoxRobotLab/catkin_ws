#!/usr/bin/env python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import time
import numpy as np
import rospy
import random
from tensorflow import keras
# import olri_classifier.olin_factory as factory #use for matchPlanner.py
import olin_factory as factory #use for olin_cnn.py
from datetime import datetime
from turtleControl import TurtleBot


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
        rospy.init_node("olin_test")
        self.robot = TurtleBot()
        self.robot.pauseMovement()


        cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path).item()
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
        print("*** Model restored: ", factory.paths.checkpoint_name)
        self.model.summary()

        self.recent_n_max=recent_n_max

    def clean_image(self, image):
        #resized_image = cv2.resize(image, (factory.image.size, factory.image.size))
        image = cv2.resize(image, (170, 128))
        x = random.randrange(0, 70)
        y = random.randrange(0, 28)
        cropped_image = image[y:y + 100, x:x + 100]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        submean_image = np.subtract(gray_image, self.mean)
        cleaned_image = np.array([submean_image], dtype="float") \
            .reshape(1, factory.image.size, factory.image.size, 1)
        return cleaned_image

    def test_turtlebot(self):
        # cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path).item()
        # intlabel_to_cell_dict = dict()
        # for cell in cell_to_intlabel_dict.keys():
        #     intlabel_to_cell_dict[cell_to_intlabel_dict[cell]] = cell
        # print(intlabel_to_cell_dict)
        lastpreds = []
        predValues = []
        potentialHeadings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        while (not rospy.is_shutdown()):
            turtle_image, _ = self.robot.getImage()

            if len(lastpreds) <= 10:
                lastpreds.append(self.get_prediction(turtle_image))
            else:
                lastpreds.pop(0)
                lastpreds.append(self.get_prediction(turtle_image))

            print(potentialHeadings[max(set(lastpreds), key=lastpreds.count)])
            predValues.append(self.get_prediction(turtle_image)[0])
            print("mean:", np.mean(predValues))
            print("std:", np.std(predValues))

            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if (ch == "q"):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

    def get_prediction(self, image):
        cleaned_image = self.clean_image(image)
        pred = self.model.predict(cleaned_image)

        sorted_pred = np.argsort(pred)[0][-3:][::-1] #top 3 best matches, with best match at index 0


        #pred_class = np.argmax(pred[153:])
        best_preds = []
        # for i in sorted_pred:
        #     pred_cell = self.intlabel_to_cell_dict[i]
        #     best_preds.append(pred_cell)

        #pred_cell = self.intlabel_to_cell_dict[int(pred_class)]
        #print("PRED HEADING: "+ pred_class)
        # best_cell = self.mode_from_recent_n(self.recent_n_max, pred_cell)

        return pred[0][np.argmax(pred[0][:])], np.argmax(pred[0][:])

if __name__ == "__main__":
    test = OlinTest(5)
    test.test_turtlebot()
