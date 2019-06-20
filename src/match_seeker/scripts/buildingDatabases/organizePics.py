# coding=utf-8
''' renames files in consecutive order and moves them from one folder to another. It also writes a txt file cataloging
the old name of every image and its new name. This is very useful if you have deleted a lot of files throughout the
database or you need to combine two smaller groups of images into one. Make sure to update the paths at the top and the
 startNum on line 61. startNum is equal to 0 if you’re renumbering a standalone directory, or to the highest image
 number in the folder you want to move your pictures into (which will become the name of the first file in your current
 directory, since images index at 0). Images do not have to be named in exact consecutive order, but it is important
 that there are no duplicate names to avoid overwriting the database. NOTE: If you are not combining image folders, you
 need to set your images to move into an empty folder. Be careful when moving images into a folder that already has
 images in it, because this program does not keep a copy of the original folder and it is not easily undone. If you mess
  up you may need to rerun image deletion and renaming on multiple old versions of your images.

Modified by Jinyoung Lim, Malini Sharma, Jane Pellegrini
Date: Jun 4, 2018
Acknowledgement: Used many functions from copyMarkedFiles.py by Susan Fox
'''

import cv2
import shutil
import os

#TODO: MAKE SURE TO CHANGE startNewNum Everytime you run it (or just change the code to make it more user-friendly)

class OrganizeData(object):
    def __init__(self, origFramesPath, desFramesPath, origLocFilePath, desLocFilePath, nameChangeFile, startNewNum=0):
        self.origFramesPath = origFramesPath
        self.desFramesPath = desFramesPath
        self.origLocFilePath = origLocFilePath
        self.desLocFilePath = desLocFilePath
        self.startNewNum = startNewNum  # starting frame number for the merged file in desFramesPath and desLocFile

        self.outLocFile = open(self.desLocFilePath, 'a')

        self.locData = self._readLocationData()

        self.nameChange = dict()
        self.nameChangeFile = open(nameChangeFile, "a")
        # Record original frames' path every time this program runs to help users figure out whats from where
        self.nameChangeFile.write(self.origFramesPath)
        self.nameChangeFile.write('\n')


    def makeFilename(self, fileNum):
        """Makes a filename for reading or writing image files"""
        formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
        name = formStr.format(self.origFramesPath,
                              'frame',
                              fileNum,
                              "jpg")
        return name

    #TODO: makeFilename and makeNewFilename could be merged
    def makeNewFilename(self, fileNum):
        """Makes a filename for reading or writing image files"""
        formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
        name = formStr.format(self.desFramesPath,
                              'frame',
                              fileNum,
                              "jpg")
        return name


    def _getImageFilenames(self):
        """Read filenames in folder, and keep those that end with jpg or png  (from copyMarkedFiles.py)"""
        filenames = os.listdir(self.origFramesPath)
        filenames.sort()
        keepers = []
        for name in filenames:
            if name.endswith("jpg") or name.endswith("png"):
                keepers.append(name)
        return keepers


    def _readLocationData(self):
        """Reads in the location file, building a dictionary that has the image number as key, and the location
        as its value.  (from copyMarkedFiles.py)"""
        locDict = dict()
        locFile = open(self.origLocFilePath, 'r')
        for line in locFile:
            if line[0] == '#' or line.isspace():
                continue
            parts = line.split()
            imageNum = int(parts[0])
            pose = [float(x) for x in parts[1:]]
            locDict[imageNum] = pose
        locFile.close()
        return locDict


    def _dataToString(self, imgNum, pose):
        lineTemplate = "{0:d} {1:d} {2:f} {3:f} {4:d}\n"
        return lineTemplate.format(imgNum, int(pose[0]), pose[1], pose[2], int(pose[3]))


    def _extractNum(self, fileString):
        """Finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for c in fileString:
            if c in '0123456789':
                foundDigits = True
                numStr += c
            elif foundDigits:
                break
        if numStr != "":
            return int(numStr)
        else:
            return -1


    def go(self):
        newNum = self.startNewNum

        for num in self.locData.keys():
            fileName = self.makeFilename(num)
            newFileName = self.makeNewFilename(newNum)
            self.nameChange[num] = newNum

            poseData = self.locData[num]
            self.outLocFile.write(self._dataToString(newNum, poseData))

            if not os.path.isdir(self.desFramesPath):
                os.mkdir(self.desFramesPath)
            shutil.copy2(fileName, newFileName)
            newNum += 1

        self.nameChangeFile.close()
        self.outLocFile.close()
        print("Finished merging with the last newNum " + str(newNum-1) + ". PLEASE change startNewNum for the next run to " + str(newNum) + ".")


if __name__ == "__main__":
    #TODO: Check for "/" at the end of directory paths

    #TODO: Always check for the correct startNewNum (usually by going into name changes file.
    startNewNum = 95838
    organizer = OrganizeData(origFramesPath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/06-18-2019-15-58-47newframes/",
                             desFramesPath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/moreframes_temp/",
                             origLocFilePath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/frameHeadingFile06-19-2019-11-21-30.txt",
                             desLocFilePath="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/THE_MASTER_CELL_LOC_244-256_renum.txt",
                             nameChangeFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morenamechanges229.txt",
                             startNewNum=startNewNum)
    organizer.go()
