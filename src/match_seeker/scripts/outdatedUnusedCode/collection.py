""" ========================================================================
collection.py

This makes a collection of ImageFeature objects from the images retrieved from
the path provided.

======================================================================== """



import os
from DataPaths import basePath
import cv2


currDirectory = basePath + "res/refinedFeb2017Data/"
baseName = "frames"
extension = "jpg"



def makeCollection():
    """Reads in all the images in the specified directory, start number and end number, and
    makes a list of ImageFeature objects for each image read in."""
    if (currDirectory is None):
        print("ERROR: cannot run makeCollection without a directory")
        return


    listDir = os.listdir(currDirectory)

    for file in listDir:
        pic = cv2.imread(currDirectory + file)
        end = len(file)-(len(extension) + 1)
        picNum = int(file[len(baseName):end])
        # cv2.imshow("pic", pic)
        # cv2.waitKey(0)


        # features = ImageFeatures.ImageFeatures(image, picNum, self.logger, self.ORBFinder)
        # self.featureCollection[picNum] = features

        # if i % 100 == 0:
        #     print i


makeCollection()










