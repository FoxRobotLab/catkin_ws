import numpy as np
import cv2

from Tracker import *

class TargetScanner(object):
    def __init__(self, object_name, frame_dims):
        """Creates the object and generates the histogram for the object to be tracked (object_name)."""
        self.object_name = object_name
        self.fWidth, self.fHeight = frame_dims
        self.full_track_window = (0,0,self.fWidth,self.fHeight)
        

        # Only direct references to the files seemed to work here
        try:
            self.object_image = cv2.imread("/home/macalester/catkin_ws/src/test_movement/scripts/images/" + object_name + ".jpg")
        except:
            print object_name + " image could not be read"
            
        self.averageColor = self.calcAverageColor()        
        
        object_hsv = cv2.cvtColor(self.object_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(object_hsv, np.array((0., 60., 32.)), np.array((255., 255., 255.)))
        self.hist = cv2.calcHist([object_hsv], [0], mask, [16], [0,180])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)             
        
        #Contains trackers for objects already found by the searcher
        self.tracking = []
        
        #Tracker that is looking for new objects
        self.searcher = Tracker(self.full_track_window, False)
        
    
    def calcAverageColor(self):
            """Calculates the average color of the object to be tracked."""
            imgWidth, imgHeight, imgDepth = self.object_image.shape
            area = imgWidth * imgHeight
            return (int(np.sum(self.object_image[:,:,0]) / area),
                    int(np.sum(self.object_image[:,:,1]) / area),
                    int(np.sum(self.object_image[:,:,2]) / area))     
        
        
    def scan(self, frame, hsv_frame, mask):
        """Updates all the of trackers for identified objects and updates the searcher which is looking for new objects."""
        bproj = cv2.calcBackProject([hsv_frame], [0], self.hist, [0,180], 1)        
        bproj &= mask
        
        for index, tracker in enumerate(self.tracking):
            original_bproj = bproj.copy()
            box, bproj, split = tracker.update(bproj)
            
            if split:
                self.splitTracker(tracker)
                del self.tracking[index]
                bproj = original_bproj

            if tracker.hasFound():
                cv2.ellipse(frame, box, self.averageColor, 2)
            else:
                del self.tracking[index]
                
        box, bproj, split = self.searcher.update(bproj.copy())
        
        if split:
            self.splitTracker(self.searcher)
            self.searcher = Tracker(self.full_track_window, found = False)
        
        if self.searcher.hasFound():
            self.tracking.append(self.searcher)
            self.searcher = Tracker(self.full_track_window, found = False)
            
        return frame

    
    def getTrackingInfo(self):
        """Returns the tracking info ((coords),area) about all the currently identified objects."""
        info = []
        area = 1.0 * self.fWidth * self.fHeight + 1.0

        for tracker in self.tracking:
            if tracker.hasFound():
                relativeArea = tracker.getArea() / area
                info.append((tracker.getCenter(), relativeArea))
        return info
        
    
    def splitTracker(self, tracker):
        """Using the info from tracker, it splits the tracker into 4 new trackers with each having a trackbox of the 4 quadrants of the original tracker."""
        c,r,w,h = tracker.getTrackWindow()
        
        w1 = w // 2
        w2 = w - w1
        h1 = h // 2
        h2 = h - h1
        
        for newBox in [(c, r, w1, h1), (c+w1+1, r, w2, h1), (c, r+h1+1, w1, h2), (c+w1+1, r+h1+1, w2, h2)]:
            self.tracking.append(Tracker(newBox, True))


