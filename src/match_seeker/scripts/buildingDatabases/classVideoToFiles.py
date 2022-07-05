
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
    capt = cv2.VideoCapture(videoName)
    frameNum = 0
    while True:
        res, frame = capt.read()
        if not res:
            break
        cv2.imshow("frames", frame)
        cv2.waitKey(10)
        saveToFolder(frame, frameFolder, frameNum)
        frameNum += 1
    capt.release()
    # cv2.destroyAllWindows()


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




