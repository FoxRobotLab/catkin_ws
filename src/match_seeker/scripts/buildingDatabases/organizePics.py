'''Rename and put pictures from different folders into one'''



from src.match_seeker.scripts.OSPathDefine import basePath
import cv2
import sys
import shutil
import os


origDir = "/home/macalester/catkin_ws/signpics/"
desDir = "/home/macalester/catkin_ws/src/match_seeker/res/kobukiSigns/"
nameChange = open("nameChangesCreateJun06.txt", "w")
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
i=704

print origDir
print desDir
print picNumList[0:2]


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
