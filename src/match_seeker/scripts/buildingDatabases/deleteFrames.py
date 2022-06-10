import os

""" ------------------------------------------------------------------------------------------------------------------
File: deleteFrames.py
Authors: Jane Pellegrini, Jinyoung Lim
Date: May 2018

Simply goes through a text file that is created by cullPictures and has names of frames to be removed line by line to
delete them without much pain in the wrists.
------------------------------------------------------------------------------------------------------------------"""

class DeleteFrames(object):
    def __init__(self, imageDirPath, deleteTextFileName):
        self.imageDirPath = imageDirPath
        self.deleteTextFileName = deleteTextFileName
        self.toBeDeleted = set()    # Prevents duplicate names (possible before fixing cullPictures)

    def readDeleteTextLines(self):
        file = open(self.deleteTextFileName, "r")
        while True:
            line = file.readline()
            line = line.rstrip()
            if (line.startswith("f")):  # Hard coding of determining if the line's name is "frames<frameNum>"
                self.toBeDeleted.add(line)
            if not line: break

    def deleteFrames(self):
        self.readDeleteTextLines()
        counter = 0
        for frameName in self.toBeDeleted:
            filePath = self.imageDirPath + frameName
            # Check if the file exists: https://stackoverflow.com/questions/8933237/how-to-find-if-directory-exists-in-python
            if (os.path.exists(filePath)):
                os.remove(self.imageDirPath + frameName)
                counter += 1
            else:
                print("Path " + filePath + " does NOT exist!")

        print("Deleted from " + self.imageDirPath + " total of " + str(counter))

if __name__ == "__main__":
    deleter = DeleteFrames(imageDirPath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/juneElevenFrames/",    #do NOT forget to put "/" at the end!
                           deleteTextFileName="/home/macalester/juneElevenToDelete.txt")
    deleter.deleteFrames()
