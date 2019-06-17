#!/usr/bin/env python


"""
speaker.py

The program you call on the laptop of the robot to receive
info from the base station and speak through the laptop's speaker.

"""


import rospy
from std_msgs.msg import String
from espeak import espeak

def callback(data):
    espeak.set_voice("english-us", gender=1, age=600)
    rospy.loginfo(data.data)
    print data.data
    espeak.synth(data.data)


def listener():
    rospy.init_node('listener',anonymous=True)
    rospy.Subscriber("chatter",String,callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
