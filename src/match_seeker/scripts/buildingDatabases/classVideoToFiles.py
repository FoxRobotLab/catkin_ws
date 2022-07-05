
import os
import time
import datetime
import cv2


def saveVideo(destDir):
    """Takes in the name of a video file and the name of a file classifying the video frames into
    categories, and it saves frames from the video into separate folders based on classification."""
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    videoName = destDir + '/' + timestamp + ".avi"
    frameFolder = destDir + '/' + timestamp + 'frames/'
    os.mkdir(frameFolder)
    # TODO: Get a frame from somewhere so we can ask about its size
    (hgt, wid, _) = frame.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(videoName, fourcc, 25.0, (wid, hgt), 1)  # TODO: Check these parameters
    frameNum = 0
    counter = 0
    while True:
        frame = # Get frame from ROS here
        vidWriter.write(frame)
        cv2.imshow("frames", frame)
        cv2.waitKey(10)
        counter += 1
        if counter == 10:
            saveToFolder(frame, frameFolder, frameNum)
            frameNum += 1
            counter = 0
    vidWriter.release()



def saveToFolder(img, folderName, frameNum):
    fName = nextFilename(frameNum)
    pathAndName = folderName + fName
    try:
        cv2.imwrite(pathAndName, img)
    except:
        print("Error writing file", frameNum, pathAndName)


def nextFilename(num):
    fTempl = "frames{0:04d}.jpg"
    fileName = fTempl.format(num)
    return fileName




