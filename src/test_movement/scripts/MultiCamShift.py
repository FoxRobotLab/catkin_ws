import numpy as np
import cv2
import threading
import math

from TargetScanner import *

class MultiCamShift(threading.Thread):
    
    def __init__(self, exampleImage):
        """Creates the cam shift thread and sets up scanners for all objects listed in 'self.toScanFor'. Needs an example image to get the dimensions of the frame."""

        threading.Thread.__init__(self)
        self.toScanFor = ["purple", "green"]
        self.scanners = {}        
        self.lock = threading.Lock()
        self.locationAndArea = {"purple" : [], "green" : []}
        
        if exampleImage == None:
            cap = cv2.VideoCapture(0)
            ret, exampleImage = cap.read()
            cap.release()
        
        self.fWidth, self.fHeight, self.fDepth = exampleImage.shape
        for object_name in self.toScanFor:
            self.scanners[object_name] = TargetScanner(object_name, (self.fWidth, self.fHeight))
        
        
    def runWithVideoFrom(self, vid_src):
        """Will run the tracking program on the video from vid_src."""
       
        self.vid = cv2.VideoCapture(vid_src)
        ret, frame = self.vid.read()
        
        cv2.namedWindow("MultiTrack")
            
        while(ret):
            self.update(frame)
 
            cv2.imshow("MultiTrack", frame)
            
            char = chr(cv2.waitKey(50) & 255)
            if char == "0":
                break

            ret,frame = self.vid.read()

        self.close()


    def update(self, image):
        """Updates the trackers with the given image."""

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, np.array((0., 60., 32.)), np.array((255., 255., 255.)))       

        objects = {}
        
        for object_name in self.scanners:
            scanner = self.scanners[object_name]
            image = scanner.scan(image, hsv_image, mask)
            objects[object_name] = scanner.getTrackingInfo()
        
        with self.lock:
            self.locationAndArea = objects

        return image
        
        
    def getObjectsOfColor(self, color_name):
        """Returns a list of objects locations and area of all identified objects of type 'color_name'."""

        with self.lock:
            locationAndArea = self.locationAndArea[color_name]
        return locationAndArea
    
    
    def getAverageOfColor(self, color_name):
        """Returns the average location and sum of area of all identified objects of type 'color_name'."""

        with self.lock:
            dataList = self.locationAndArea[color_name]
        xTotal = 0
        yTotal = 0
        aTotal = 0.0
        for data in dataList:
            ((x,y), a) = data
            xTotal += x
            yTotal += y
            aTotal += a
        size = len(dataList)
        return [((xTotal / size, yTotal / size), aTotal)] if size != 0 else []

    
    def close(self):
        """Closes the program."""

        cv2.destroyAllWindows()
        self.vid.release()
