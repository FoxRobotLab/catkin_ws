# !/usr/bin/env python

""" ========================================================================
  turtleControl.py

   Created on: June 2016

   The turtleControl.py file borrows code from TurtleBot.py in speedy_nav.
  Four classes: one for the TurtleBot object, and then three thread classes -
  MovementControlThread, ImageSensorThread, and SensorThread.  TurtleBot
  initialises all the threads and provides movement/utility methods -
  turning, bumper status, find angle to wall, and so on.
  Uses Utils. Uses CvBridge to be able to use images from kinect with OpenCV/ROS

========================================================================="""


import rospy
import threading
import sys
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError
import tf
from tf import transformations
import numpy
from Utils import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from create_node.msg import TurtlebotSensorState
from kobuki_msgs.msg import SensorState
from std_msgs.msg import Empty
from time import time


class TurtleBot(object):
    """This is a robot object that communicates with the three separate ROS-communicating threads, and it provides a
    set of methods that is easier to use (modelled on Pyro/Myro commands) than working with the threads directly."""

    def __init__(self):
        """Sets up the three threads and starts them running."""

        self.robotType = os.environ["TURTLEBOT_BASE"]

        if self.robotType == "create":
            self.degreeToSeconds = 0.046
            self.angleTurnSpeed = 0.4
        elif self.robotType == "kobuki":
            self.degreeToSeconds = 0.039
            self.angleTurnSpeed = 0.5

        self.moveControl = MovementControlThread(self.robotType)
        self.moveControl.start()

        self.depthControl = DepthSensorThread(self.robotType)
        self.depthControl.start()

        self.imageControl = ImageSensorThread(self.robotType)
        self.imageControl.start()

        self.odom = OdometryListener(self.robotType)
        self.odom.start()
        self.odom.resetOdometer()

        rospy.on_shutdown(self.exit)


    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""
        turnSec = abs(angle * self.degreeToSeconds)
        if angle > 0:
            self.turnLeft(self.angleTurnSpeed, turnSec)
        elif angle < 0:
            self.turnRight(self.angleTurnSpeed, turnSec)
        else:
            # No need to turn, keep going
            pass


    def findAngleToWall(self):
        """This looks at the depth data and compares the left edge of the depth image with the right edge of the depth image,
        computing the angle of the robot to the wall (probably not that accurate, but it's something)."""
        width, height = self.depthControl.getDims()
        sectionWidth = width / 30.0
        sectionHeight = height / 5.0
        # box = c,r,w,h
        leftBox = int(sectionWidth * 10), int(sectionHeight * 2), int(sectionWidth), int(sectionHeight)
        rightBox = int(sectionWidth * 19), int(sectionHeight * 2), int(sectionWidth), int(sectionHeight)

        leftDist = numpy.mean(self.depthControl.getDepth(*leftBox))
        rightDist = numpy.mean(self.depthControl.getDepth(*rightBox))
        return getMedianAngle(leftDist, rightDist, angleBetween = 11.0)

    def forward(self, amount, seconds = None):
        """Takes in an amount, which is a speed, and an optional input, seconds, which is a number of seconds. If no
        seconds are given, then it sets the robot to go forward indefinitely (or until superceded by a new movement
        command). If seconds are given, then it has the robot make a timed movement to move forward for the specified time."""
        if seconds is None:
            self.moveControl.setMovement(amount, 0.0)
        else:
            self.moveControl.timedMovement(amount, 0.0, seconds)

    def backward(self, amount, seconds = None):
        """Takes in an amount, which is a speed, and an optional input, seconds, which is a number of seconds. If no
        seconds are given, then it sets the robot to go backward indefinitely (or until superceded by a new movement
        command). If seconds are given, then it has the robot make a timed movement to move backward for the specified time."""
        if seconds is None:
            self.moveControl.setMovement(-amount, 0.0)
        else:
            self.moveControl.timedMovement(-amount, 0.0, seconds)

    def turnLeft(self, amount, seconds = None):
        """Takes in an amount, which is a speed, and an optional input, seconds, which is a number of seconds. If no
         seconds are given, then it sets the robot to turn left indefinitely (or until superceded by a new movement
         command). If seconds are given, then it has the robot make a timed movement to turn left for the specified time."""
        if seconds is None:
            self.moveControl.setMovement(0.0, amount)
        else:
            self.moveControl.timedMovement(0.0, amount, seconds)

    def turnRight(self, amount, seconds = None):
        """Takes in an amount, which is a speed, and an optional input, seconds, which is a number of seconds. If no
         seconds are given, then it sets the robot to turn right indefinitely (or until superceded by a new movement
         command). If seconds are given, then it has the robot make a timed movement to turn right for the specified time."""
        if seconds is None:
            self.moveControl.setMovement(0.0, -amount)
        else:
            self.moveControl.timedMovement(0.0, -amount, seconds)

    def stop(self):
        """Stops the robot's movement."""
        self.moveControl.stopMovement()

    def translate(self, amount):
        """Takes in amount, a speed, and sets the robot's translate value to the given speed. The robot will start
        forward movement (perhaps mixed with a rotation amount set separately) and continue indefinitely."""
        self.moveControl.setTranslate(amount)

    def rotate(self, amount):
        """Takes in amount, a speed, and sets the robot's rotation value to the given speed. The robot will start
        rotating (perhaps mixed with a forward or backward movement set separately) and continue indefinitely."""
        self.moveControl.setRotate(amount)

    def move(self, translate, rotate):
        """Takes in a translation speed and a rotation speed and sets the robot to moving with both (a curving path). It
        will continue indefinitely."""
        self.moveControl.setMovement(translate, rotate)

    def getDepth(self, x = 0, y = 0, width = 640, height = 480):
        """Takes in (x, y), the upper left corner of a rectangle, and the rectangle's botWidth and botHeight, all optional
        inputs. It access the depth image and returns the specified section of the depth image. The default values are
        to return the entire depth image for the Kinect's sensor."""
        return self.depthControl.getDepth(x, y, width, height)

    def getImage(self, x = 0, y = 0, width = 640, height = 480):
        """Takes in (x, y), the upper left corner of a rectangle, and the rectangle's botWidth and botHeight, all optional
        inputs. It accesses the camera image and returns the specified section of the camera image. The default values are
        to return the entire camera image for the Kinect's sensor."""
        return self.imageControl.getImage(x, y, width, height)

    def getOdomData(self):
        return self.odom.getData()


    def updateOdomLocation(self,x = 0, y = 0, yaw = 0.0):
        """updates an offset for all the odometry data, when skews are introduced by human or environmental interference.
        has a default of 0 for all inputs, to reset the offsets."""
        return self.odom.updateOdomLoc(x, y, yaw )

    def hasWheelDrop(self):
        return self.depthControl.hasBeenWheelDrop()

    def getBumperStatus(self):
        """Accesses the robot base sensor data and reports whether the bumper has triggered."""
        state = self.depthControl.getSensorState()
        if state is None:
            return 0
        if self.robotType == "kobuki":
            return state.bumper
        else:
            return state.bumps_wheeldrops

    def getWheelDropStatus(self):
        state = self.depthControl.getSensorState()
        if state is None:
            return 0
        if self.robotType == "kobuki":
            return state.wheel_drop
        else:
            return state.bumps_wheeldrops

    def getCliffStatus(self):
        """Accesses the robot base sensor data and reports whether the cliff sensor has triggered."""
        state = self.depthControl.getSensorState()
        if state is None:
            return 0
        if self.robotType == "kobuki":
            return state.cliff
        else:
            return state.cliff_left + state.cliff_right + state.cliff_front_left + state.cliff_front_right

    def exit(self):
        """A method that shuts down the three threads of the robot."""
        print("************** RECEIVED SHUTDOWN **************")
        self.moveControl.haltRun()
        self.moveControl.join()
        self.depthControl.exit()
        self.depthControl.join()
        self.imageControl.exit()
        self.imageControl.join()
        self.odom.exit()
        self.odom.join()



class MovementControlThread(threading.Thread):
    """Thread communicates with ROS to implement movement commands."""

    def __init__(self, robotType):
        """Sets up lock and instance variables that control running, translation and rotation values,
        and a rate limiter."""
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.runFlag = True
        self.translate = 0.0
        self.rotate = 0.0
        self.rateLimiter = rospy.Rate(10)  # 10Hz
        self.paused = False
        self.robot = robotType

        if robotType == "create":
            self.move_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size = 10)
        elif robotType == "kobuki":
            self.move_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=10)

    def run(self):
        """Thread's run method, runs, getting the translation and rotation set by other methods."""
        twist = Twist()
        with self.lock:
            twist.linear.x = self.translate
            twist.angular.z = self.rotate
            self.runFlag = True
        runFlag = True
        paused = False
        try:
            while runFlag:
                with self.lock:
                    paused = self.paused
                if not paused:
                    self.move_pub.publish(twist)
                    self.rateLimiter.sleep()
                    with self.lock:
                        twist.linear.x = self.translate
                        twist.angular.z = self.rotate
                        runFlag = self.runFlag
        except Exception, e:
            print e
        finally:
            twist = Twist()  # default to no motion
            self.move_pub.publish(twist)

    def setRotate(self, amount):
        """Method typically called by other threads, to set the rotation amount."""
        with self.lock:
            self.rotate = amount

    def setTranslate(self, amount):
        """Method typically called by other threads, to set the translation amount."""
        with self.lock:
            self.translate = amount

    def setMovement(self, translate, rotate):
        """Method typically called by other threads, to set both translation and rotation amounts."""
        with self.lock:
            self.translate = translate
            self.rotate = rotate

    def stopMovement(self):
        """Method typically called by other threads, to stop the robot's movement."""
        with self.lock:
            self.translate = 0.0
            self.rotate = 0.0

    def timedMovement(self, translate, rotate, seconds):
        """Method typically called by other threads, to set movement for a fixed number of seconds."""
        with self.lock:
            self.paused = True
        try:
            twist = Twist()
            twist.linear.x = translate
            twist.angular.z = rotate
            r = rospy.Rate(10)  # 10Hz
            for i in range(int(seconds * 10)):
                self.move_pub.publish(twist)
                r.sleep()
        except Exception, e:
            print e
        finally:
            twist = Twist()  # default to no motion
            self.move_pub.publish(twist)
            with self.lock:
                self.paused = False  # start up the continuous motion loop again

    def haltRun(self):
        """Method typically called by other threads, to stop the robot and shut down this thread."""
        with self.lock:
            self.runFlag = False
            self.translate = 0.0
            self.rotate = 0.0


class ImageSensorThread(threading.Thread):
    """This thread communicates with the camera data from ROS, providing updated camera images upon request."""

    def __init__(self, robotType):
        """Creates the connections to ROS, sets up CvBridge to translate ROS images to OpenCV ones, and initializes
        the thread."""
        threading.Thread.__init__(self)
        print "Image sensor thread init"
        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.imageProcessed = False
        self.image_array = None
        self.savedImage = None
        self.numTimesImgServed = 0

        self.robotType = robotType

        if self.robotType == "create":
            self.image_sub = rospy.Subscriber("/camera/image_decompressed", Image, self.image_callback)
        elif self.robotType == "kobuki":
            self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color",Image,self.image_callback)

        self.runFlag = True

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
        self.image_sub.unregister()

    def image_callback(self, data):
        """Callback function connected to ROS camera data, gets called whenever the camera object
        produces any data. Next time an image is requested, it will be processed."""
        with self.lock:
            self.image_array = data
            self.imageProcessed = False

    def getImage(self, x, y, width, height):
        """Method typically called by other threads, gets the next image available. If no images are available yet,
        this method blocks until one becomes available."""

        with self.lock:
            data = self.image_array
            processed = self.imageProcessed
        if not processed:
            try:
                if data is None:
                    sys.stdout.write("Waiting for camera data")
                    while data is None:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        rospy.sleep(0.2)
                        if rospy.is_shutdown():
                            return
                        with self.lock:
                            data = self.image_array
                    print " Done!"
                cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            except CvBridgeError, e:
                print e

            retval = cv_image[y:y + height, x:x + width]
            if self.robotType == "kobuki":
                r, g, b = cv2.split(retval)
                retval = cv2.merge((b,g,r))
            with self.lock:
                self.savedImage = retval
                self.imageProcessed = True
            self.numTimesImgServed = 1
        else:
            with self.lock:
                retval = self.savedImage
            self.numTimesImgServed += 1
        return retval, self.numTimesImgServed

    def exit(self):
        """Method typically called by other threads, to shut down this thread."""
        print "Image sensor shutdown received"
        with self.lock:
            self.runFlag = False


class DepthSensorThread(threading.Thread):
    """This thread communicates with the ROS depth data, and also the mobile_base sensors (like the bumper and onboard IR sensor."""

    def __init__(self, robotType):
        """Sets up the thread, connects to the ROS depth data (with a callback) and to the ROS sensor data with a callback."""
        print "Depth sensor thread init"
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.robotType = robotType
        self.depth_array = None
        self.sensor_state = None

        if self.robotType == "create":
            self.depth_sub = rospy.Subscriber("/camera/depth_decompressed", Image, self.depth_callback)
            self.sensor_sub = rospy.Subscriber("/mobile_base/sensors/core", TurtlebotSensorState, self.sensor_callback)
        elif self.robotType == "kobuki":
            self.depth_sub = rospy.Subscriber("/camera/depth/image_raw",Image,self.depth_callback)
            self.sensor_sub = rospy.Subscriber("/mobile_base/sensors/core", SensorState, self.sensor_callback)

        self.runFlag = True
        self.wheelFlag = False


    def run(self):
        """Thread's run method. Does nothing, as the work is done either by callbacks being triggered or by other threads requesting data."""
        with self.lock:
            self.runFlag = True
        runFlag = True

        while runFlag:
            rospy.sleep(0.1)
            with self.lock:
                runFlag = self.runFlag
        self.sensor_sub.unregister()
        self.depth_sub.unregister()

    def sensor_callback(self, data):
        """Callback function triggered when sensor data is available. Just copies to instance variable."""
        with self.lock:
            self.sensor_state = data
        if data.wheel_drop>0:
            self.wheelFlag = True
        # print self.sensor_state

    def hasBeenWheelDrop(self):
        with self.lock:
            hasDrop = self.wheelFlag
            self.wheelFlag = False
        return hasDrop


    def depth_callback(self, data):
        """Callback function triggered when depth data is available, Just copies data to instance variable."""
        with self.lock:
            self.depth_array = data


    def getDims(self):
        """Method typically called by another thread, it returns the botWidth and botHeight of the depth data, if available,
        or (0, 0) if it is not yet ready."""
        with self.lock:
            data = self.depth_array
        if data is None:
            return 0, 0
        return data.width, data.height

    def getSensorState(self):
        """Method typically called by another thread, it returns the robot base sensor state."""
        with self.lock:
            state = self.sensor_state
        return state

    def getDepth(self, x, y, width, height):
        """Method typically called by another thread, it takes in integers x, y, botWidth, and botHeight, which specify a
        rectangular region of the depth image, and it returns a depth image or the specified portion of it.
        If no depth image data is available, this method blocks until the data becomes available."""
        with self.lock:
            data = self.depth_array
        try:
            if data is None:
                sys.stdout.write("Waiting for depth data")
                while data is None:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    if rospy.is_shutdown():
                        return
                    rospy.sleep(0.1)
                    with self.lock:
                        data = self.depth_array
                print " Done!"
            cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
            print e
        numpy_array = numpy.asarray(cv_image)
        retval = numpy_array[y:y + height, x:x + width]
        return retval


    def exit(self):
        """Method typically called by another thread, it triggerse the shutdown of this thread."""
        print "Depth sensor shutdown received"
        with self.lock:
            self.runFlag = False

class OdometryListener(threading.Thread):
    """This thread communicates with the robot's odometry node, providing updated odometry data upon request."""


    def __init__(self, robotType):
        """Creates the connections to ROS, and initializes
        the thread."""
        threading.Thread.__init__(self)
        print "Odometry listener thread init"
        self.lock = threading.Lock()
        self.odomSub = rospy.Subscriber("odom", Odometry, self._odomCallback)
        self.runFlag = True
        self.robotType = robotType

        self.x = None
        self.y = None
        self.yaw = None
        self.offsetX = 22.2
        self.offsetY = 6.5
        self.offsetYaw = 0.0


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
            #converts weird quarternian data points from the robots orientation to radians.
            (roll, pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x,
                                                                                data.pose.pose.orientation.y,
                                                                                data.pose.pose.orientation.z,
                                                                                data.pose.pose.orientation.w])
            #roll and pitch are only useful if your robot can move in the z plane. our turtlebots cannot fly.
            self.yaw = math.degrees(self.yaw)

    def updateOdomLoc(self, x, y, yaw):
        """Lets you offset the odometry data, in case of outside input to the robots location,
        like a chair or some helping hands."""
        self.offsetX = x - self.x
        self.offsetY = y - self.y
        self.offsetYaw = yaw - self.yaw
        print "Offsets: ", str(self.offsetX), str(self.offsetY), str(self.offsetYaw)
        return self.offsetX, self.offsetY, self.offsetYaw

    def getData(self):
        """Method typically called by other threads, gets the available odometry data. If no data is available yet,
        this method blocks until some becomes available."""

        with self.lock:
            x, y, yaw = self.x, self.y, self.yaw
        return x + self.offsetX, y+self.offsetY, yaw + self.offsetYaw

    def resetOdometer(self):
        if self.robotType == 'kobuki':
            # set up the odometry reset publisher
            reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=10)

            # reset odometry (these messages take a few iterations to get through)
            timer = time()
            while time() - timer < 1.0: #was 0.25
                reset_odom.publish(Empty())
        else:
            pass

    def exit(self):
        """Method typically called by other threads, to shut down this thread."""
        print "Odometry shutdown received"
        with self.lock:
            self.runFlag = False


if __name__ == "__main__":

    rospy.init_node('test_movement')
    controller = TurtleBot()
    rospy.on_shutdown(controller.exit)
    rospy.spin()
