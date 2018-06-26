from gtk.keysyms import x
import cv2
import src.match_seeker.scripts.turtleControl
import time


class turtlebotVideoStraightToFrames(object):

    def __init__(self, outputFolder, outputFile):

        self.outputFile = outputFile
        self.outputFolder = outputFolder

        self.robot = src.match_seeker.scripts.turtleControl.TurtleBot()

        self.frameNum = 0
        self.currentTime = 0

    def saveFrames(self):
        ch= ""
         # ch = chr(x & 0xFF) #this line is meant to convert key commands (waitKey) into strings. So I put it after
         #waitkey instead of here
        file = open(self.outputFile, 'w')

        while ch != 'q':
            currImg = self.robot.getImage()
            self.saveToFolder(currImg)
            file = open(self.outputFile, 'w')
            self.saveToTextFile(file)
            self.frameNum = self.frameNum + 1
            x= cv2.waitKey(10)
            ch = chr(x & 0xFF)
        file.close()
        cv2.destroyAllWindows()


    def saveToFolder(self, img):
        fName = self.getFilename()
        pathAndName = self.outputFolder + fName
        try:
            cv2.imwrite(pathAndName, img)
        except:
            print("Error writing file", self.frameNum, pathAndName)


    def saveToTextFile(self, file):
        time1 = time.localtime()
        self.currTime = time.strftime("%H:%M:%S", time1)
        file.write("frame."+str(self.frameNum)+ ".jpg " + "time:" + str(self.currTime)+"\n") #writes the frame as well as the time it is recorded


# I am not using this at the moment because I'm not sure what it does and I want it to match the text file perfectly
    # def nextFilename(self, num):
    #     fTempl = "frame{0:04d}.jpg"
    #     fileName = fTempl.format(num)
    #     return fileName
    #

    def getFilename(self):
        fileName = "frame" + str(self.frameNum) + ".jpg"
        return fileName

if __name__ == "__main__":

    framer = turtlebotVideoStraightToFrames( outputFolder="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/",
        outputFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames.txt")

    framer.saveFrames()






