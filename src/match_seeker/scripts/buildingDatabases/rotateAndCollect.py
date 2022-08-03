#!/usr/bin/env python2.7
"""--------------------------------------------------------------------------------
rotateAndCollect.py
Authors: Bea Bautista, Yifan Wu

This program simply pops up a window waiting for commands to move the robot

To run this file:
1.Cutie must be launched (minimal, astra, and teleop)
2. In terminal run command: q


To end this program (if it does not end as expected),
processes must be killed in the terminal using kill -9 <Python2.7 PID>
--------------------------------------------------------------------------------"""

import rospy
import time
#sys.path.append('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts') # handles weird import errors
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf
import cv2
import numpy as np
#from pynput import keyboard

# uncomment to use CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

class CollectRotation:

    def __init__(self):
        rospy.init_node('rotate', anonymous=True, disable_signals=True)

        self.move_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=10)
        self.odomSub = rospy.Subscriber("odom", Odometry, self.odomCallback)

        self.turnFlagR = False
        self.turnFlagL = False
        self.forwardFlag = False
        self.backwardFlag = False
        self.amount = 0.1
        #self.step = 40
        # self.forwardCertainDistanceFlag = False

        self.offsetX = 6.1  # 20.0
        self.offsetY = 41.1  # 26.2
        self.offsetYaw = 270.0  # 90.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0
        self.degreeToSeconds = 0.039
        self.angleTurnSpeed = 0.2
        self.twist = Twist()

        self.prevX, self.prevY, self.prevYaw = self.getData()

    def turnByAngle(self, angle):
        """Turns the robot by the given angle, where negative is left and positive is right"""

        currX, currY, currHead = self.getData()
        goalHead = currHead + angle
        if goalHead > 180:
            goalHead -= 360
        elif goalHead < -180:
            goalHead += 360

        while (currHead > goalHead + 5 or currHead < goalHead - 5) and (self.turnFlagL or self.turnFlagR):

            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)
            if ch =="s" \
                    "":
                self.turnOffFlags()

            if angle > 0:
                self.turnLeft(self.angleTurnSpeed)
            elif angle < 0:
                self.turnRight(self.angleTurnSpeed)
            else:
                # No need to turn, keep going
                pass

            currHead = self.getData()[2]
            time.sleep(0.2)
        self.stopMovement()

    def turnRight(self, amount):
        """Takes in an amount, which is a speed, then it sets the robot to turn right indefinitely (or until superceded by a new movement
         command)."""

        self.twist.linear.x = 0
        self.twist.angular.z = -amount
        self.move_pub.publish(self.twist)

    def turnLeft(self, amount):
        self.twist.linear.x = 0
        self.twist.angular.z = amount
        self.move_pub.publish(self.twist)

    def forward(self, amount):
        self.twist.linear.x = amount
        self.twist.angular.z = 0
        self.move_pub.publish(self.twist)

    def backward(self, amount):
        self.twist.linear.x = -amount
        self.twist.angular.z = 0
        self.move_pub.publish(self.twist)

    def stopMovement(self):
        """Stops the robot's movement."""

        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.move_pub.publish(self.twist)



    def getData(self):
        """Gets the available odometry data. If no data is available yet,
        this method blocks until some becomes available."""
        x, y, yaw = self.x, self.y, self.yaw

        newYaw = yaw + self.offsetYaw
        if newYaw > 180:
            newYaw -= 360
        elif newYaw < -180:
            newYaw += 360

        return x + self.offsetX, y+self.offsetY, newYaw


    def odomCallback(self, data):
        radHead = math.radians
        self.x = data.pose.pose.position.x # * math.cos(radHead) - data.pose.pose.position.y * math.sin(radHead)
        self.y = data.pose.pose.position.y # * math.sin(radHead) + data.pose.pose.position.y * math.cos(radHead)

        # converts weird quarternian data points from the robots orientation to radians.
        (roll, pitch, self.yaw) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x,
                                                                            data.pose.pose.orientation.y,
                                                                            data.pose.pose.orientation.z,
                                                                            data.pose.pose.orientation.w])
        # roll and pitch are only useful if your robot can move in the z plane. our turtlebots cannot fly.
        self.yaw = math.degrees(self.yaw)

    # def on_press(self, key):
    #     try:
    #         self.k = key.char  # single-char keys
    #     except:
    #         self.k = key.name  # other keys
    #     if key == 'q':
    #         return False  # stop listener

    def turnOffFlags(self):
        self.turnFlagR = False
        self.turnFlagL = False
        self.forwardFlag = False
        self.backwardFlag = False


    def mainThread(self):
        blank_image = 255 * np.ones(shape=[300 ,1200 , 3], dtype=np.uint8)

        firstline = "a: left turn 360 || d: right turn 360"
        secondline = "w: move forward indefinitely || x: move backward indefinitely"
        thirdline = "z: keep hitting for moving left || c: keep hitting for moving right"
        fourthline = "q: quit || s: stop"

        turtle_image_text = cv2.putText(img=blank_image, text=firstline, org=(50, 50), fontScale=1,
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0,0,0))
        turtle_image_text = cv2.putText(img=turtle_image_text, text=secondline, org=(50, 80), fontScale=1,
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))
        turtle_image_text = cv2.putText(img=turtle_image_text, text=thirdline, org=(50, 110), fontScale=1,
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))
        turtle_image_text = cv2.putText(img=turtle_image_text, text=fourthline, org=(50, 140), fontScale=1,
                                        fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0))

        cv2.imshow("Turtle Image", turtle_image_text)

        # listener = keyboard.Listener(on_press=self.on_press)
        # listener.start()  # start to listen on a separate thread

        while not rospy.is_shutdown():

            key = cv2.waitKey(10)
            ch = chr(key & 0xFF)

            if ch == "a":
                self.turnOffFlags()
                self.turnFlagL = True
                i = 0

            if ch == "w":
                self.turnOffFlags()
                self.forwardFlag = True

            if  ch == "d":
                self.turnOffFlags()
                self.turnFlagR = True
                i = 0

            if ch =="z":
                self.turnOffFlags()
                self.turnLeft(self.angleTurnSpeed)
                time.sleep(0.2)

            if ch =="x":
                self.turnOffFlags()
                self.backwardFlag = True

            if ch =="c":
                self.turnOffFlags()
                self.turnRight(self.angleTurnSpeed)
                time.sleep(0.2)

            if ch == "s":
                self.turnOffFlags()
                self.stopMovement()

            if self.forwardFlag:
                self.forward(self.amount)
                time.sleep(0.2)

            if self.backwardFlag:
                self.backward(self.amount)
                time.sleep(0.2)

            # if self.forwardCertainDistanceFlag:
            #     self.forward(self.amount)
            #     time.sleep(0.2)
            #
            #     i += 1
            #     if i == self.step:
            #         self.forwardCertainDistanceFlag = False

            if self.turnFlagR:
                self.turnByAngle(-45)

                i += 1
                if i%2 == 0:
                    time.sleep(2)
                if i == 8:
                    self.turnByAngle(-20)
                    self.turnOffFlags()

            if self.turnFlagL:
                self.turnByAngle(45)

                i += 1
                if i%2 == 0:
                    time.sleep(2)
                if i == 8:
                    self.turnByAngle(20)
                    self.turnOffFlags()


            if ch == "q":
                cv2.destroyAllWindows()
                self.move_pub.unregister()
                self.odomSub.unregister()
                print("Robot shutdown start")
                rospy.signal_shutdown("End")
                print("Robot shutdown complete")
                #listener.join()  # remove if main thread is polling self.keys

if __name__ == "__main__":
    rotater = CollectRotation()
    try:
        rotater.mainThread()
    except rospy.ROSInterruptException:
        pass
