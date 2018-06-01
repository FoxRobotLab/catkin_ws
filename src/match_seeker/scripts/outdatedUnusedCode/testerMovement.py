#!/usr/bin/env python

"""This is a little tester function to test movements of the robot"""


import turtleControl
import MovementHandler
import OutputLogger
import rospy


class testerMovement(object):

    def __init__(self):
        self.robot = turtleControl.TurtleBot()
        self.logger = OutputLogger.OutputLogger(True, True)
        self.move = MovementHandler.MovementHandler(self.robot, self.logger)


    def turn(self):
        while True:
            angle = raw_input("Please type in the desired angle. ")
            if angle != "q":
                self.robot.turnByAngle(int(angle))
            else:
                break

    def sec(self):
        while True:
            turnFor = raw_input("time to turn (sec): ")
            if turnFor != 'q':
                sec = float(turnFor)
                self.robot.turnRight(0.5,sec)
            else:
                break

    def odom(self):
        i = 0
        while True:
            i +=1
            if i==20:
                i = 0
                print self.robot.getOdomData()


if __name__ == "__main__":
    rospy.init_node('Tester')
    tester = testerMovement()
    tester.turn()
