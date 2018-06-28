import turtleControl
import time
import cv2
import os
import rospy

class StraightToFrames(object):

    def __init__(self, outputFolder, outputFile, robot):

        self.picNum = 0000

        self.dictOfTimes = dict()
        self.currTime = 0
        self.currTime2 = 0

        self.outputFolder = outputFolder
        self.outputFile = outputFile
        self.robot = robot
        self.img = None

    def go(self):
        ch = ''
        while ch != 'q':
            print("Starting while loop")
            self.img, _ = self.robot.getImage()
            cv2.imshow("Image", self.img)
            time.sleep(1)
            self.currTime = time.localtime()
            self.currTime2 = time.strftime("%H:%M:%S", self.currTime)
            self.picNum = self.picNum + 1
            self.dictOfTimes[self.picNum] = self.currTime2
            self.saveToFolder(self.img, self.outputFolder, self.picNum)
            x = cv2.waitKey(10)
            ch = chr(x & 0xFF)


        self._writeData()


    def saveToFolder(self, img, folderName, frameNum):
            fName = self.nextFilename(frameNum)
            pathAndName = os.path.join(folderName + fName)
            try:
                cv2.imwrite(pathAndName, img)
            except:
                print("Error writing file", frameNum, pathAndName)

    def _writeData(self):
        """Write the data collected to a timestamped file."""
        try:
            os.makedirs(self.outputFile)
        except:
            pass
        logName = self.outputFile
        print(logName)
        fileOpen = False
        logFile = None
        try:
            logFile = open(self.outputFile, 'w')
            fileOpen = True
        except:
            print ("FAILED TO OPEN DATA FILE")

        for key in self.dictOfTimes:
            time = self.dictOfTimes[key]
            dataStr = str(key) + " " + str(time) + "\n"
            if fileOpen:
                logFile.write(dataStr)
            print("Frame", key, "at time", time)
        logFile.close()

    def nextFilename(self, num):
        fTempl = "frame{0:04d}.jpg"
        fileName = fTempl.format(num)
        return fileName

    def _setupWindows(self):
        "Creates three windows and moves them to the right places on screen."
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 30, 550)


def main():
    rospy.init_node('StraightToFrames')
    robot = turtleControl.TurtleBot()

    framer = StraightToFrames(
        outputFolder='/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/',
        outputFile='/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/testTurtlebotVidFrames.txt',
        robot=robot)
    framer.go()


if __name__ == "__main__":
    main()
