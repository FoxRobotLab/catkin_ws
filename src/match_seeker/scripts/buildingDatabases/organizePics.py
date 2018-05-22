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
'''

from src.match_seeker.scripts.DataPaths import basePath
import cv2
import sys
import shutil
import os


origDir = "/home/macalester/catkin_ws/src/match_seeker/res/homeSigns/"
desDir = "/home/macalester/catkin_ws/src/match_seeker/res/kobuki0615/"
nameChange = open("kobukiSignsNameChanges.txt", "w")
nameChange.write(origDir)
nameChange.write('\n')

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
    name = formStr.format(origDir,
                          'frame',
                          fileNum,
                          "jpg")
    return name



def makeNewFilename(fileNum):
    """Makes a filename for reading or writing image files"""
    formStr = "{0:s}{1:s}{2:0>4d}.{3:s}"
    name = formStr.format(desDir,
                          'frame',
                          fileNum,
                          "jpg")
    return name



listDir = os.listdir(origDir)
picNumList = []

#gets list of every file in the directory in order
for file in listDir:
    end = len(file) - (len('jpg') + 1)
    picNum = int(file[len('frame'):end])
    picNumList.append(picNum)

picNumList.sort()

#starting number
i=1619

for num in picNumList:
    fileName = makeFilename(num)
    newFileName = makeNewFilename(i)
    nameChange.write(str(i) + ' ' + str(num) + '\n')
    shutil.move(fileName, newFileName)
    i += 1

nameChange.close()
# file = "/home/macalester/catkin_ws/src/match_seeker/res/test/origName.txt"
# shutil.copy(file,desDir)
# shutil.move(file, desDir + "rename.txt")
# shutil.move(file,"/home/macalester/catkin_ws/src/match_seeker/res/test/")
