#!/usr/bin/env python
import turtleControl
import time
import cv2
import datetime
import rospy
import os

#"""Author: Malini Sharma"""


currDate = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
os.mkdir(
    '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/' + currDate + 'newframes/')


class StraightToFrames(object):

    def __init__(self, outputFolder, outputFileName, robot):

        """Initializes variables needed to run the program. These variables include the sequential number of pictures,
        a dictionary with the picture number as the key and the time it was obtained as its value, the current time in
        terms of local time, the current time when changed into the format we want for the text file, the path for the
        the output folder for the pictures, the path for the output text file, a robot object and the given image"""

        self.picNum = 0000

        self.dictOfTimes = dict()
        self.currTime = 0
        # self.currTime2 = 0

        self.currCell = 217

        self.outputFolder = outputFolder
        self.outputFileName = outputFileName
        self.robot = robot

        self.robot.pauseMovement()

        self.img = None

    def go(self):

        """This function makes the class run. The while loop runs until the user presses Q and rospy is not shutdown
        This function gets the image from the turtlebot robot object, displays the image for the user, and waits a
        second. It then gets the time via the current time and converts it into the format we need for the text file.
        Next, the picture number is incremented and the picture number and time are added to the dictionary and the
        image is save to the output folder. The last part of the while loop is setting up the character input. Outside of
        the while loop, this function writes the picture number and time data into the text file, closes the image
        window, and shuts down rospy."""

        ch = ''

        while ch != 'q' and not rospy.is_shutdown():
            self.img, _ = self.robot.getImage()
            cv2.imshow("Image", self.img)
            time.sleep(.5)
            #self.currTime = time.localtime()
            #self.currTime2 = time.strftime("%H:%M:%S", self.currTime)
            self.currTime = int(time.time() * 1000)
            #self.currTime = datetime.datetime.strftime(self.currTime, "%H:%M:%S")
            self.picNum = self.picNum + 1
            self.dictOfTimes[self.picNum] = [self.currTime, self.currCell]
            saved = self.saveToFolder(self.img, self.outputFolder, self.picNum)
            x = cv2.waitKey(10)
            ch = chr(x & 0xFF)
            if ch =='n':
                print("you're changing loc")
                self.currCell-=1
            if not saved:
                break
        self._writeData()
        cv2.destroyAllWindows()
        rospy.signal_shutdown("shutting down")

    def saveToFolder(self, img, folderName, frameNum):

        """This function saves an image to the output folder."""

        fName = self.nextFilename(frameNum)
        pathAndName = folderName + fName
        try:
            works= cv2.imwrite(pathAndName, img)
            if not works:
                print("Error writing file! QUIT!", frameNum, pathAndName)
                return False
        except:
            print("Error writing file", frameNum, pathAndName)
            return False
        return True

    def _writeData(self):

        """This function writes the data in the dictionary to the output file."""

        fileOpen = False
        logFile = None
        textFileName = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/' + currDate + 'newframes/' + currDate+ 'frames.txt' #os.path.join(self.outputFolder, self.outputFileName)    #Assumes self.outputFileName is just a standalone file name
        try:
            logFile = open(textFileName, 'w')
            fileOpen = True
        except:
            print ("FAILED TO OPEN DATA FILE")

        for key in self.dictOfTimes:
            time = self.dictOfTimes[key][0]
            cell = self.dictOfTimes[key][1]
            dataStr = str(key) + " " + str(time) + "\n"
            if fileOpen:
                logFile.write(dataStr)
            print("Frame", key, "at time", time)
        logFile.close()

    def nextFilename(self, num):

        """This function is a helper function for the writeData() function. It gives writeData() the next file name."""


        fTempl = "frame{0:04d}.jpg"
        fileName = fTempl.format(num)
        return fileName


def main():
    rospy.init_node('StraightToFrames')

    robot = turtleControl.TurtleBot()

    framer = StraightToFrames(
        outputFolder='/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/' + currDate + 'newframes/',
        outputFileName='june10frames.txt',
        robot=robot)
    framer.go()


if __name__ == "__main__":
    main()

# '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/turtlebotVideoStraightToFrames.py/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/turtlebotVideoStraightToFrames.py'
