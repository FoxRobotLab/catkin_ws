# coding=utf-8


"""
shows user the matching images from the dataset Flags for removal with user permission.

 uses the same image matching program as the program to find matching images in the dataset. If you took video, there
 will be dozens of frames saved at every location, and you just don’t need to have that many images in the database.

 This program displays two images. If they match, press y to record the number of the matching image in a txt file.
 Press any other key to advance.

 NOTE this program assumes that the first image in a series of matching images is the
 best one, but that is often not the case since the camera image gets very blurry during turning. We found it easiest
 to have pen and paper nearby either while doing this or just after and recording blurry image numbers while passing over
 them during deletion. You can then add your list of blurry images to the list of matches compiled by this program and
 delete them all at once.
"""

# import src.match_seeker.scripts.ImageFeatures
import cv2
import numpy
from src.match_seeker.scripts.DataPaths import basePath


class scanForDeletes(object):

    def __init__(self,
                 direct=None,
                 baseName='frame', ext="jpg",
                 resultFile = "picsToDelete.txt",
                 startPic=0, numPics=-1):
        self.currDirectory = direct
        self.baseName = baseName
        self.currExtension = ext
        self.startPicture = startPic
        self.numPictures = numPics
        self.resultFile = resultFile
        self.threshold = 800.0
        self.height = 0
        self.width = 0



        print "Length of collection = " + str(self.numPictures)

    def markDeletes(self):
        """displays a picture and the one before it, and the diffs, marking the evaluation score. Then user types
        'd' to delete the picture, and any other key to go on to the next one. The lists of marked for deletion pictures
        are saved to the given filename."""
        if (self.currDirectory is None) \
            or (self.numPictures == -1):
            print "ERROR: cannot run cycle without at least a directory and a number of pictures"
            return

        delNums = []
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 350, 50)
        for i in range(self.numPictures):
            picNum = self.startPicture + i
            image = self.getFileByNumber(picNum)
            if image is None:
                continue
            cv2.imshow("Image", image)
            x = cv2.waitKey(0)
            ch = chr(x & 0xFF)
            if ch == 'd':
                delNums.append(picNum)
            elif ch == 'q':
                break

        writFile = open(self.resultFile, 'w')
        for num in delNums:
            writFile.write(str(num) + '\n')
        writFile.close()





    def getFileByNumber(self, fileNum):
        """Makes a filename given the number and reads in the file, returning it."""
        filename = self.makeFilename(fileNum)
        image = cv2.imread(filename)
        # if image is None:
        #     print("Failed to read image:", filename)
        return image

    def putFileByNumber(self, fileNum, image):
        """Writes a file in the current directory with the given number."""
        filename = self.makeFilename(fileNum)
        cv2.imwrite(filename, image)

    def makeFilename(self, fileNum):
        """Makes a filename for reading or writing image files"""
        formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
        name = formStr.format(self.currDirectory,
                              self.baseName,
                              fileNum,
                              self.currExtension)
        return name


if __name__ == '__main__':
    delScanner = scanForDeletes(
        direct="/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/res/May2417/",
        baseName="frame",
        ext="jpg",
        resultFile="testingDelList.txt",
        startPic=0,
        numPics=1000)
    # scan.makeCollection()
    delScanner.markDeletes()
