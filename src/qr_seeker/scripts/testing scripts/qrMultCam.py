#!/usr/bin/env python

import cv2
import numpy as np
import zbar
from PIL import Image

def qrScan(image):
    qrScanner = zbar.ImageScanner()
    qrScanner.parse_config('enable')
    bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(bwImg)
    pic2 = pil_im.convert("L")
    wid, hgt = pic2.size
    raw = pic2.tobytes()

    img = zbar.Image(wid, hgt, 'Y800', raw)
    result = qrScanner.scan(img)
    #print "RESULT", result
    if result == 0:
        print "Scan failed"
        return None
    else:
        #print ("img is ", img)
        for symbol in img:
            #print "symbol did indeed get assigned"
            pass
        del(img)
        codeData = symbol.data.decode(u'utf-8')
        print "Data found:", codeData

def getNextFrame(vidObj):
    ret, frame = vidObj.read()
    return frame

def initCam(cam, framerate, width, height):
    cam.set(cv2.CAP_PROP_FPS, framerate)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def onboardCamera():
    framerate = 30
    width = 480
    height = 360
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
        qrScan(frame1)
        qrScan(frame2)
        cv2.imshow("camshift 1", frame1)
        cv2.imshow("camshift 2", frame2)
        keypress = chr(cv2.waitKey(10) & 255)
        if keypress == "t":
            cv2.imwrite("/home/macalester/Desktop/githubRepositories/catkin_ws/src/qr_seeker/res/refs/blue_webcam.jpg", vis2) 
            print "Image saved."
        elif keypress == "q":
            break

onboardCamera()
