#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from espeak import espeak

def callback(data):
    rospy.loginfo(data.data)

def listener():
    rospy.init_node('listener',anonymous=True)

    str = rospy.Subscriber("chatter",String,callback)

    espeak.synth(str)

    rospy.spin()


if __name__ == '__main__':
    listener()
