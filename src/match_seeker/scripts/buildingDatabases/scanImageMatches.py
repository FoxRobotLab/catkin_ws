# coding=utf-8
"""
shows user the matching images from the dataset. Flags for removal with user permission.

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

import src.match_seeker.scripts.ImageFeatures
import cv2
import numpy
from src.match_seeker.scripts.DataPaths import basePath


class scanImageMatches(object):

    def __init__(self,
                 dir1=None, dir2=None,
                 baseName='foo', ext="jpg",
                 startPic=0, numPics=-1):
        self.currDirectory = dir1
        self.secondDirectory = dir2
        self.baseName = baseName
        self.currExtension = ext
        self.startPicture = startPic
        self.numPictures = numPics
        self.threshold = 800.0
        self.height = 0
        self.width = 0

        cv2.ocl.setUseOpenCL(False)
        self.ORBFinder = cv2.ORB_create()
        self.featureCollection = {}



    def makeCollection(self):
        """Reads in all the images in the specified directory, start number and end number, and
        makes a list of ImageFeature objects for each image read in."""
        print "in make collection"
        if (self.currDirectory is None) \
            or (self.numPictures == -1):
            print "ERROR: cannot run makeCollection without a directory and a number of pictures"
            return
        print "Reading in image database"

        for i in range(self.numPictures):
            picNum = self.startPicture + i
            image = self.getFileByNumber(picNum)
            if self.height == 0:
                self.height, self.width, depth = image.shape
            features = src.match_seeker.scripts.ImageFeatures.ImageFeatures(image, picNum, None, self.ORBFinder)
            self.featureCollection[picNum] = features
            if i % 100 == 0:
                print i

        print "Length of collection = " + str(self.numPictures)


    def compare(self):
        """Compares all to all. Must have called makeCollection before calling this operation."""

        if (self.currDirectory is None) \
            or (self.numPictures == -1):
            print "ERROR: cannot run cycle without at least a directory and a number of pictures"
            return
        elif len(self.featureCollection) == 0:
            print "ERROR: must have collection built before cycling"
            return
        print "Comparing all pictures with all"

        matchScore = {}
        matchNums = []
        cv2.namedWindow("image 1")
        cv2.namedWindow("match")
        cv2.namedWindow("DIFF")
        cv2.moveWindow("image 1", 50, 50)
        cv2.moveWindow("match", 700, 50)
        cv2.moveWindow("DIFF", 1400, 50)
        for i in range(self.numPictures):
            picNum = self.startPicture + i
            newFirst = True
            if len(matchNums) is 0:
                newFirst = False
            features1 = self.featureCollection[picNum]


            if picNum not in matchNums:
                for j in range(i+1, self.numPictures):  # Note this duplicates for debugging purposes comparisons both ways
                    pic2num = self.startPicture + j
                    if pic2num not in matchNums:
                        features2 = self.featureCollection[pic2num]
                        simVal = features1.evaluateSimilarity(features2)
                        matchScore[picNum, pic2num] = simVal
                        # logStr = "image " + str(picNum) + " matched image " + str(pic2num) + " with score " + str(simVal)
                        # print logStr
                        if simVal <= 10:
                            if newFirst:
                                blank = numpy.zeros((self.height, self.width), numpy.uint8)
                                cv2.putText(blank, str(picNum), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
                                cv2.imshow("image 1", blank)
                                cv2.waitKey(0)
                                newFirst = False
                            img1 = self.getFileByNumber(picNum)
                            img2 = self.getFileByNumber(pic2num)
                            diffIm = cv2.absdiff(img1, img2)
                            cv2.putText(img1,str(picNum),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0))
                            cv2.putText(img2, str(pic2num), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0))
                            cv2.imshow("image 1", img1)
                            cv2.imshow("match", img2)
                            cv2.imshow("DIFF", diffIm)
                            response = cv2.waitKey(0)
                            response = chr(response%0x255)
                            if response == 'y':
                                matchNums.append(pic2num)
        print matchNums
        matchFile = open('dupKobuki052918_atriumClockwise_1550_1749.txt', 'w')
        for item in matchNums:
            matchFile.write("%s\n" % item)
        matchFile.close()


    def getFileByNumber(self, fileNum):
        """Makes a filename given the number and reads in the file, returning it."""
        filename = self.makeFilename(fileNum)
        image = cv2.imread(filename)
        if image is None:
            print("Failed to read image:", filename)
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
    scan = scanImageMatches(
                           dir1 = "/home/macalester/turtlebot_videos/atriumClockwiseFrames/",
                           baseName = "frames",
                           ext = "jpg",
                           startPic = 1550,
                           numPics = 200)
    scan.makeCollection()
    scan.compare()


























