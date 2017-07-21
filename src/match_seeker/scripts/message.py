#!/usr/bin/env python


"""
message.py

The program you call on the base station, and will send info
to the robot's laptop to speak if speak.py is on on the robot.

"""

import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        str = raw_input("Say something. ")
        rospy.loginfo(str)
        pub.publish(str)
        rate.sleep()


def speak(phrase):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    str = phrase
    rospy.loginfo(str)
    pub.publish(str)


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

