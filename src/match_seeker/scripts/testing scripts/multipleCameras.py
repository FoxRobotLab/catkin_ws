#!/usr/bin/env python

import cv2
import numpy as np


def getNextFrame(vidObj):
    ret, frame = vidObj.read()
    return frame

def initCam(cam, framerate, width, height):
    cam.set(cv2.CAP_PROP_FPS, framerate)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def onboardCamera():
    framerate = 12
    #width = 480
    #height = 360
    width = 640
    height = 480

    defCam = cv2.VideoCapture()
    r, f = defCam.read()
    if r:
        print "default"
        cv2.imshow("default", f)
        cv2.waitKey(0)
    print "starting for loop"
    for i in range(20):
        nextCam = cv2.VideoCapture(i)
        if nextCam.isOpened():
            print "Camera at", i
            frame =  getNextFrame(i)
            cv2.imshow("camera" + str(i), frame)
            cv2.waitkey(0)
        else:
            print "trying to open"
            nextCam.open()
            if nextCam.isOpened():
                print "Had to open", i
            else:
                print "Found no camera at", i

    print "for loop ended"
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)

    initCam(cam1, framerate, width, height)
    initCam(cam2, framerate, width, height)

    cv2.namedWindow('camshift 1')
    cv2.namedWindow('camshift 2')

    # start processing frames
    while True:
        frame1 = getNextFrame(cam1)
        frame2 = getNextFrame(cam2)
        cv2.imshow("camshift 1", frame1)
        cv2.imshow("camshift 2", frame2)
        keypress = chr(cv2.waitKey(10) & 255)
        if keypress == "t":
            cv2.imwrite("/home/macalester/Desktop/githubRepositories/catkin_ws/src/qr_seeker/res/refs/blue_webcam.jpg", vis2)
            print "Image saved."
        elif keypress == "q":
            break

onboardCamera()
