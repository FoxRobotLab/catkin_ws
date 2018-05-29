
import os
import cv2

def saveVideo(videoName, destDir):
    """Takes in the name of a video file and the name of a file classifying the video frames into
    categories, and it saves frames from the video into separate folders based on classification."""
    capt = cv2.VideoCapture(videoName)
    frameNum = 0
    while True:
        res, frame = capt.read()
        if not res:
            break
        #cv2.imshow("frame",frame)
        #cv2.waitKey(10)
        saveToFolder(frame, destDir, frameNum)
        frameNum+=1
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
    fTempl = "frame{0:04d}.jpg"
    fileName = fTempl.format(num)
    return fileName


saveVideo("/home/macalester/turtlebot_videos/atrium_clockwise.avi", "/home/macalester/turtlebot_videos/atriumClockwiseFrames/")

