import os
import cv2
import numpy as np

""" ------------------------------------------------------------------------------------------------------------------
File: deleteFrames.py
Date: May 2018

Simply goes through a text file that is created by cullPictures and has names of frames to be removed line by line to 
delete them without much pain.
------------------------------------------------------------------------------------------------------------------"""

class DeleteFrames(object):
    def __init__(self, imageDirPath, deleteTextFileName):
        self.imageDirPath = imageDirPath
        self.deleteTextFileName = deleteTextFileName
        self.toBeDeleted = set()    #Prevents duplicate names (possible before fixing cullPictures)

    def readDeleteTextLines(self):
        file = open(self.deleteTextFileName, "r")
        while True:
            line = file.readline()
            line = line.rstrip()
            if (line.startswith("f")):
                self.toBeDeleted.add(line)
            if not line: break

    def deleteFrames(self):
        self.readDeleteTextLines()
        for frameName in self.toBeDeleted:
            filePath = self.imageDirPath + frameName
            # Check if the file exists: https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
            if (os.path.exists(filePath)):
                os.remove(self.imageDirPath + frameName)
            else:
                print("Path " + filePath + " does NOT exist!")

        print("Deleted from " + self.imageDirPath + " total of " + str(len(self.toBeDeleted())))

if __name__ == "__main__":
    deleter = DeleteFrames(imageDirPath="/home/macalester/turtlebot_videos/atriumSouthFrames_2 (copy)/",    #do NOT forget to put "/" at the end!
                           deleteTextFileName="/home/macalester/turtlebot_videos/atriumSouthDelete_2 (copy).txt")
    deleter.deleteFrames()
