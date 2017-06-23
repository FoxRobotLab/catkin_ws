#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from espeak import espeak

def callback(data):
    espeak.set_voice("english-us", gender=2, age=10)
    rospy.loginfo(data.data)
    print data.data
    espeak.synth(data.data)


def listener():
    rospy.init_node('listener',anonymous=True)
    rospy.Subscriber("chatter",String,callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
