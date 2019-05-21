"""
turtlebot_cnn_runner.py
Author: Jinyoung Lim
Date: June 2018

Test floortype cnn using turtlebot. Move turtlebot with keyboard op. Speak and print out perceived floor type.

Acknowledgements:
    match_seeker's matchPlanner and turtlebotControl
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import rospy
import floortype_cnn
from espeak import espeak
import turtleControl
from std_msgs.msg import String
import os
import random
import numpy as np

class TurtlebotCNNRunner(object):
    def __init__(self):
        rospy.init_node("FloorClassifier")
        # print("IN MAIN \t 1 \t INIT NODE")
        # # turtle_cnn = TurtlebotCNNRunner()
        # print("IN MAIN \t 2")
        # # turtle_cnn.run()
        # print("IN MAIN \t 3")
        # # rospy.spin()
        # self.cnnmodel = floortype_cnn.getModel()

        print("IN __init__ \t 0")
        self.robot = turtleControl.TurtleBot()
        print("IN __init__ \t 1")
        # self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        # print("IN __init__ \t 2")

        self.pub = rospy.Publisher('chatter', String, queue_size=10)



    def get_turtlebot_image(self):
        image, _ = self.robot.getImage()
        # image = cv2.resize(image, (80, 80))
        return image

    def run(self):
        print("IN run \t 0")
        while not rospy.is_shutdown():
            print("IN run \t 1")
            image, _ = self.robot.getImage()
            print(image.shape)
            print(type(image))

            # cv2.imwrite("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/turtlebot_image.jpg", image)
            # savepath = floortype_cnn.process_test_data_and_return_savepath("/home/macalester/PycharmProjects/tf_floortype_classifier/turtlebot_runner_images/")
            # print(savepath)
            # test_data = np.load(savepath)
            smallImg = cv2.resize(image, (80, 80))
            pred_str = floortype_cnn.test_one(smallImg, self.cnnmodel)
            print("PRED STR: ",pred_str)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(pred_str, font, 1, 2)[0]
            text_x = int((image.shape[1] - text_size[0]) / 2)
            text_y = int((image.shape[0] + text_size[1]) / 2)

            cv2.putText(
                img=image,
                text=pred_str,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2)

            cv2.imshow("Turtlebot View with CNN Floortype", image)
            x = cv2.waitKey(20)
            ch = chr(x & 0xFF)

            if (ch == "q"):
                break
        print("IN run \t 99")
        self.robot.stop()
        print("IN run \t 100")

    def speak(self, speakStr):
        """Takes in a string and "speaks" it to the base station and also to the  robot's computer."""
        espeak.set_voice("english-us", gender=2, age=10)
        espeak.synth(speakStr)  # nodeNum, nodeCoord, heading = matchInfo
        self.pub.publish(speakStr)

if __name__ == "__main__":
    print("IN MAIN \t 0")
    rospy.init_node("FloorClassifier")
    print("IN MAIN \t 1")
    turtle_cnn = TurtlebotCNNRunner()
    print("IN MAIN \t 2")
    turtle_cnn.run()
    print("IN MAIN \t 3")
    rospy.spin()
    print("IN MAIN \t 4")