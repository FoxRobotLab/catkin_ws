#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from espeak import espeak

def callback(data):
    rospy.loginfo(data.data)
    print data.data
    espeak.synth(data.data)

def listener():
    rospy.init_node('listener',anonymous=True)

    rospy.Subscriber("chatter",String,callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
