#!/usr/bin/env python


'''This class is used to implement Monte Carlo Localization'''


import numpy as np
import cv2
from DataPaths import basePath
from markLocations import readMap
import math




class monteCarloLoc():

    def __init__(self):
        self.mapWid = 1203
        self.mapHgt = 816
        self.worldX, self.worldY = self.convertMapToWorld(self.mapWid, self.mapHgt)
        self.mapFile =  basePath + "res/map/olinNewMap.txt"

        # x1, y1, x2, y2 (x1 <= x2; y1 <= y2)
        # self.obstacles = [(0, 0, 1204, 359),   # bounding tuples of rectangles that represent classrooms, stairs, and walls
        #                    (1092, 359, 1204, 535),
        #                    (1092, 659, 1204, 1009),
        #                    (358, 409, 1058, 589),
        #                    (354, 631, 1058, 815),
        #                    (366, 815, 974, 869),
        #                    (0, 901, 1092, 1009)]

        # self.obstacles = [(0, -193, 1202, 165),
        #                   (1092, 165, 1202, 341),
        #                   (1092, 465, 1202, 815),
        #                   (358, 215, 1058, 395),
        #                   (354, 437, 1058, 621),
        #                   (366, 621, 974, 675),
        #                   (0, 815, 1092, 707)]

        self.obstacles = [(self.convertWorldToMap(50.4, 60.1) + self.convertWorldToMap(32.5, 0.0)),
               (self.convertWorldToMap(32.5, 5.5) + self.convertWorldToMap(23.7, 0.0)),
               (self.convertWorldToMap(17.5, 5.5) + self.convertWorldToMap(0.0, 0.0)),
               (self.convertWorldToMap(30.0, 42.2) + self.convertWorldToMap(21.0, 7.2)),
               (self.convertWorldToMap(18.9, 42.4) + self.convertWorldToMap(9.7, 7.2)),
               (self.convertWorldToMap(9.7, 41.8) + self.convertWorldToMap(7.0, 11.4)),
               (self.convertWorldToMap(0.0, 60.1) + self.convertWorldToMap(5.4, 5.5))]

        print self.obstacles

        self.validPosList = []
        for i in range(10):
            self.addParticle()

    def addParticle(self):
        """generating a new list with random possibility all over the map"""

        posAngle = np.random.uniform(-2*np.pi + np.pi/2, 2*np.pi + np.pi/2)  # radians :P
        # posAngle = math.radians(0) + math.pi/2
        # posX = 22.2
        # posY = 6.5
        posX, posY = np.random.uniform(0, self.mapWid, 2)
        posX = int(posX)
        posY= int(posY)
        # self.validPosList.append((posX, posY, posAngle))
        if self.isValid((posX,posY)):
            self.validPosList.append((posX, posY, posAngle))
        else:
            self.addParticle()



    def particleMove(self, moveX, moveY, moveAngle):
        """updating the information of the points when the robot moves"""
        for posPoint in self.validPosList:
            posPoint[0] += moveX
            posPoint[1] += moveY
            posPoint[2] += moveAngle
        self.update(self.validPosList)



    def update(self, posList):
        """Updating the possibility list and removing the not valid ones"""
        updatedList = []

        for particle in posList:
            if self.isValid(particle):
                updatedList.append(particle)

        self.validPosList = updatedList

        # Adding randomized particles to the map
        for i in range(4):
            self.addParticle()


    def isValid(self, posPoint):
        """checking if the particle is within the obstacle areas
        return false if it's no longer a valid point"""
        posX = posPoint[0]
        posY = posPoint[1]

        for rect in self.obstacles:
            if (posX >= rect[0] and posX <= rect[2]) and (posY >= rect[1] and posY <= rect[3]):
                if posX >= rect[0] and posX <= rect[2]:
                    print "x out of bounds"
                    print posX, rect[0], rect [2]
                    return False
                elif posY >= rect[1] and posY <= rect[3]:
                    print "y out of bounds"
                    print posY, rect[1], rect[3]
                    return False

        print "within bounds", posX, posY
        return True



    def getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(self.mapFile, 20)
        map2 = np.flipud(origMap)
        self.olinMap = np.rot90(map2)
        (self.mapHgt, self.mapWid, self.mapDep) = self.olinMap.shape


    def drawLocOnMap(self, x, y, heading):
        """Draws the current location on the map"""
        nextMapImg = self.olinMap.copy()
        self.drawPosition(nextMapImg, x,y, heading, (255, 0, 0))

        #printing stuff
        # positionTemplate = "({1:5.2f}, {2:5.2f}, {3:f})"
        # offsetTemplate = "(Offsets: {0:5.2f}, {1:5.2f}, {2:f})"
        # posInfo = positionTemplate.format(self.startX, self.startY, self.startYaw)
        # offsInfo = offsetTemplate.format(offsetX, offsetY, offsetYaw)
        # cv2.putText(nextMapImg, posInfo, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(nextMapImg, offsInfo, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        return nextMapImg


    def drawPosition(self, image, x, y, heading, color):
        cv2.circle(image, (x, y), 6, color, -1)

        line = 10
        newX = int(x + (line * math.cos(heading)))
        newY = int(y - (line * math.sin(heading)))

        cv2.line(image, (x, y), (newX, newY), color)

    def drawBlank(self):
        width = self.mapWid
        blank_image = np.zeros((width, width, 3), np.uint8)
        return blank_image

    def getParticles(self):
        return self.validPosList

    def convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY
        # Next convert to meters from pixels, assuming 20 pixels per meter
        mapXMeters = flipX / 20.0
        mapYMeters = flipY / 20.0
        return mapXMeters, mapYMeters


    def convertWorldToMap(self, worldX, worldY):
        """Converts coordinates in meters in the world to integer coordinates on the map
        Note that this also has to adjust for the rotation and flipping of the map."""
        # First convert from meters to pixels, assuming 20 pixels per meter
        pixelX = worldX * 20.0
        pixelY = worldY * 20.0
        # Next flip x and y values around
        mapX = self.mapWid - 1 - pixelY
        mapY = self.mapHgt + 194 - 1 - pixelX
        return int(mapX), int(mapY)



test = monteCarloLoc()
test.getOlinMap()
map = test.olinMap.copy()
blank = test.drawBlank()
list = test.getParticles()
blank[194:194+map.shape[0], 0:map.shape[1]] = map
for rect in test.obstacles:
    cv2.rectangle(blank,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),thickness=2)
oob = 0

print len(list)
for part in list:
    (x, y, head) = part
    # test.drawPosition(map,x,y,head,(0,0,255))
    test.drawPosition(blank,x,y,head,(0,0,255))

im = cv2.resize(blank,(900,900), interpolation = cv2.INTER_AREA)
# cv2.imshow("drawing", map)
cv2.imshow("points", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
