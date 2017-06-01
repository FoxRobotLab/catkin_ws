import rospy
from geometry_msgs.msg import Twist
import sys

twist = Twist()

def values():
    print '(w for forward, a for left, s for reverse, d for right,k for turning left,l for turning right and . to exit)' + '\n'
    s = raw_input(':- ')
    if s[0] == 'w':
        twist.linear.x = 10.0
        twist.angular.z = 0.0
        # twist.linear.y = 0.0
    elif s[0] == 's':
        twist.linear.x = -10.0
        twist.angular.z = 0.0
        # twist.linear.y = 0.0
    # elif s[0] == 'd':
    #     twist.linear.y = -1.0
    #     twist.angular.z = 0.0
    #     twist.linear.x = 0.0
    # elif s[0] == 'a':
    #     twist.linear.y = 1.0
    #     twist.angular.z = 0.0
    #     twist.linear.x = 0.0
    elif s[0] == 'k':
        twist.angular.z = 2.0
        twist.linear.x = 0
        # twist.linear.y = 0.0
    elif s[0] == 'l':
        twist.angular.z = -2.0
        twist.linear.x = 0
        # twist.linear.y = 0.0
    elif s[0] == '.':
        twist.angular.z = twist.linear.x = 0
        # twist.linear.y = 0.0
        sys.exit()
    else:
        twist.linear.x =  twist.angular.z = 0.0
        # twist.linear.y =0
        print 'Wrong command entered \n'
    return twist

def keyboard():
    pub = rospy.Publisher('mobile_base/command/velocity',Twist, queue_size=10)
    rospy.init_node('teleop_py',anonymous=True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        twist = values()
        pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        keyboard()
    except rospy.ROSInterruptException:
        pass