#!/usr/bin/env python


import rospy
from nav_msgs.msg import Odometry
import threading
import tf
from tf import transformations
from std_msgs.msg import Empty
from time import time



class odometryListener(threading.Thread):
    """This thread communicates with the odometry data from ROS, providing updated data upon request."""


    def __init__(self):
        """Creates the connections to ROS and initializes the thread."""
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.odomSub = rospy.Subscriber("odom", Odometry, self._odomCallback)
        self.runFlag = True

        self.x = None
        self.y = None
        self.yaw = None


    def run(self):
        """The thread's run method.  Does nothing active, just keeps running so that the other methods
        may be accessed from other threads."""
        with self.lock:
            self.runFlag = True
        runFlag = True
        while runFlag:
            rospy.sleep(0.2)
            with self.lock:
                runFlag = self.runFlag


    def _odomCallback(self, data):
        """Callback function connected to ROS odometry data, gets called whenever the odometry object
        produces any data."""
        with self.lock:
            self.x = data.pose.pose.position.x
            self.y = data.pose.pose.position.y
            #converts weird quarternian data points from the robots orientation to radians. make them degrees later pls
            (roll, pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x,
                                                                                data.pose.pose.orientation.y,
                                                                                data.pose.pose.orientation.z,
                                                                                data.pose.pose.orientation.w])
            # roll and pitch are only useful if your robot can move in the z plane. our turtlebots cannot fly.


    def getData(self):
        """Method typically called by other threads, gets the next available data. If no odometry data is available yet,
        this method blocks until some becomes available."""

        with self.lock:
            x, y, yaw = self.x, self.y, self.yaw
        return x, y, yaw



    def exit(self):
        """Method typically called by other threads, to shut down this thread."""
        print "odometry shutdown received"
        with self.lock:
            self.runFlag = False



class odometer():

    def __init__(self):
        self.odom = odometryListener()


    def getOdomData(self):
        return self.odom.getData()


    def exit(self):
        self.odom.exit()


    def resetOdometer(self):
        # set up node
        # rospy.init_node('reset_odom')

        # set up the odometry reset publisher
        reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=10)

        # reset odometry (these messages take a few iterations to get through)
        timer = time()
        while time() - timer < 0.25:
            reset_odom.publish(Empty())
