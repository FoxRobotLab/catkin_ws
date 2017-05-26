
import cv2
import shutil

basePath = "/home/macalester/catkin_ws/src/match_seeker/"

srcDir = basePath + "res/May2517-office2/"
desDir = basePath + "res/052517/"
editDir = open("editDirMay25-office1.txt",'r')




def getFileByNumber(fileNum):
    """Makes a filename given the number and reads in the file, returning it."""
    filename = makeFilename(fileNum)
    image = cv2.imread(filename)
    if image is None:
        print("Failed to read image:", filename)
    return image


def makeFilename(fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(srcDir,
                          'frame',
                          fileNum,
                          "jpg")
    return name

def newFilename(fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(desDir,
                          'frame',
                          fileNum,
                          "jpg")
    return name

for line in editDir:
    changeList = line.split()
    oldFile = makeFilename(int(changeList[1]))
    newFileName = newFilename(int(changeList[0]))
    shutil.move(oldFile, newFileName)





















