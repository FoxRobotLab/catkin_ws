""" ========================================================================
 * ORBrecognizer.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 *  The ORBrecognizer object computes the similarity of objects using the ORB
 *  feature detector.
 *
========================================================================="""


import cv2
import numpy as np
from operator import itemgetter


class ORBrecognizer():
    """Holds data about ORB keypoints found in the input picture."""
    def __init__(self, bot):
        self.orb = cv2.ORB_create()
        self.matches = None
        self.goodMatches = None
        self.robot = bot
        self.camSample, self.kinectSample = self.initRefs()
        self.bf = cv2.BFMatcher()
        itemsSought = ['sign', 'lastNames', 'secondString', 'exitRef', 'hatchRef', 'doorRef']
        self.properties = self.loadRefs()

    def initColorRefs(self):
        filenames = ['blue.jpg','blue_webcam.jpg']
        refs = []
        for i in range (0, 2):
            filename = filenames[i]
            """Yeah, hardcoded paths are gross and we all hate them. But ROS doesn't set the . directory
            to the current file, and making the path dynamic was significantly more effort than we wanted
            to invest. (See <http://wiki.ros.org/rospy_tutorials/Tutorials/Makefile>.) Since (due to the 
            webcams) you can't run the code remotely, you're going to be on the turtlebot laptop anyway.
            """
            path = "/home/macalester/Desktop/githubRepositories/catkin_ws/src/qr_seeker/res/refs/" + filename
            try:
                colorSample = cv2.imread(path)
                refs.append(colorSample)
            except:
                print "Could not read the sample color image " + filename + "!"
        return refs

    def tryToMatchFeatures(self, orb, img1, pointInfo, img2):
        kp2, des2 = pointInfo

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)

        if des1 is None or des2 is None:
            return [], None, None
        elif len(des1) == 0 or len(des2) == 0:
            print("it thinks the key descriptions are empty")
            return [], None, None

        # BFMatcher with default params
        matches = self.bf.match(des1,des2)

        sortedMatches = sorted(matches, key = lambda x: x.distance)
        goodMatches = [mat for mat in matches if mat.distance < 300]

        matchImage = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches,
            None, matchColor = (255, 255, 0), singlePointColor=(0, 0, 255))
        cv2.imshow("Match Image", matchImage)
        cv2.waitKey(20)

        return goodMatches, kp1, kp2


    def findImage(self, img, properties, itemsSought, whichCam):

        colorImage = img
        answer = False

        #properties[i][2] holds the list of good points and the two sets of keypoints (for drawing)
        for i in range (0, len(itemsSought)):
            goodPoints, targetKeypts, refKeypts = self.tryToMatchFeatures(self.orb, img, properties[i][1], properties[i][0])
            properties[i][2].append(goodPoints)    #[i][2][0]
            properties[i][2].append(targetKeypts)  #[i][2][1]
            properties[i][2].append(refKeypts)     #[i][2][2]

            #properties[i][3] holds the len(goodPoints) - the number of good points of that item
            numGoodPoints = len(properties[i][2][0])
            properties[i][3] = numGoodPoints

        #find the index and value of the image (which i) with the greatest number of good keypoints
        getcount = itemgetter(3)
        scores = map(getcount, properties)
        max_index, max_value = max(enumerate(scores), key=itemgetter(1))

        if (max_value > 25 and whichCam == 'center') or max_value > 40:
            print('The '+ str(itemsSought[max_index]) + ' sign was detected, with ' + str(max_value) + ' points')
            #returns the best set of good points (for the image and reference) 
            retVal = (properties[max_index][2][1], properties[max_index][2][0]) 
        else:
            retVal = None

        #cleanup
        for i in range (0, len(itemsSought)):
            properties[i][2] = []
            properties[i][3] = 0

        return retVal

    def colorPreprocessing(self, img, colorSample):
        """Takes an image and a sample of the color it wants to look at, and then masks out everything
        but that color. Includes in the image anything surrounded by said color - i.e. the letters on
        our signs."""

        img2 = img.copy() #used to apply the final mask to, later - img is destroyed, I think

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # convert to HSV
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))   # eliminate low and high saturation values

        # access the currently selected region and make a histogram of its hue
        hsv_roi = cv2.cvtColor(colorSample, cv2.COLOR_BGR2HSV) 
        hist = cv2.calcHist( [hsv_roi], [0], None, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.reshape(-1)

        #masks out everything but the color
        prob = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        prob &= mask

        #opening removes the remnants in the background
        size = 15
        kernel = np.ones((size,size),np.uint8)
        prob = cv2.morphologyEx(prob, cv2.MORPH_OPEN, kernel)

        #thresholding to black and white removes the grey areas of noise left in the image from opening
        ret,prob = cv2.threshold(prob,127,255,cv2.THRESH_BINARY)

        #split into channels in order to apply the mask to each channel and merge - sure there's a better way to do this
        blue_channel, green_channel, red_channel = cv2.split(img)
        blue_channel &= prob
        green_channel &= prob
        red_channel &= prob
        cv2.merge((blue_channel, green_channel, red_channel), img)


        """Prob at this point has black outline around the letter themselves, which is why we need
        to do all the stuff with the contours. Blobs don't work because the white areas are too large.
        Instead, find contours, draw boxes around them, and fill in the boxes."""
        _, contours, hierarchy = cv2.findContours(prob, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #make an all-black mask
        rows = img.shape[0]
        cols = img.shape[1]
        black = np.zeros((rows, cols), dtype='uint8')

        #draw a white box on the black mask around the location of all contours in the image
        for i in range (0, len(contours)):
            contour = contours[i]
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(black, [box], 0, (255,0,0), -1)

        #apply the white-boxes mask to the original image
        blue_channel, green_channel, red_channel = cv2.split(img2)
        blue_channel &= black
        green_channel &= black
        red_channel &= black
        cv2.merge((blue_channel, green_channel, red_channel), img2)

        return img2

    def initRefs(self, itemsSought):
        properties = [] #2D array used to store info on each item: item i is properties[i]

        #properties[i][0] holds the reference image for the item
        for i in range (0, len(itemsSought)):
            properties.append([None, [], [], 0])

            #see note on hardcoded paths in orbScan().
            filename = itemsSought[i] + '.jpg'
            path = "/home/macalester/Desktop/githubRepositories/catkin_ws/src/qr_seeker/res/refs/" + filename
            properties[i][0] = cv2.imread(path, 0)
            if properties[i][0] is None:
                print("Reference image", itemsSought[i], "not found")

            #properties[i][1] holds the keypoints and their descriptions for the reference image
            keypoints, descriptions = self.orb.detectAndCompute(properties[i][0], None)
            properties[i][1].append(keypoints)
            properties[i][1].append(descriptions)

        return properties

    #run this if you wanna test the feature recognition using a still image
    """This code is built so that you can check an image against multiple features.
    However, qrPlanner really only wants you to look for one, the Fox Robotics Lab sign. As 
    a result, all of this code is a lot more complicated than it needs to be. It's more flexible
    this way -- just remember than anytime you see something like 'properties[i]', i is just 
    which image you're checking against, which is always the same - the Fox Robot Lab sign. """
    def orbScan(self, image, whichCam):
        image2 = image.copy()
        if whichCam == 'center':
            colorSample = self.kinectSample
        else:
            colorSample = self.camSample
        img = self.colorPreprocessing(image2, colorSample)
        return self.findImage(img, self.properties, itemsSought, whichCam)