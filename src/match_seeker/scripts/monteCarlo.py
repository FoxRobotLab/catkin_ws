#!/usr/bin/env python


'''This class is used to implement Monte Carlo Localization'''


import numpy as np
import cv2
from DataPaths import basePath
from markLocations import readMap
import math




class monteCarloLoc():

    def __init__(self):
        self.worldX = 41.0   # meters
        self.worldY = 61.0   # meters
        self.mapWid = 1203   # pixels
        self.mapHgt = 816    # pixels
        # self.worldX, self.worldY = self.convertMapToWorld(self.mapWid, self.mapHgt)
        print self.worldX, self.worldY
        self.mapFile =  basePath + "res/map/olinNewMap.txt"

        # x1, y1, x2, y2 (x1 <= x2; y1 <= y2)
         # bounding tuples of rectangles that represent classrooms, stairs, and walls
        self.obstacles = [(32.5, 0.0, 50.4,  60.1),   # top obstacle across classrooms
                          (23.7, 0.0, 32.5, 5.5),    # ES office more or less
                          (0.0, 0.0, 17.5, 5.5),     # Near the origin
                          (21.0, 7.2, 30.0, 42.2),   # Covers 205 and 250
                          (9.7, 7.2, 18.9, 42.4),    # Covers 258 etc
                          (7.0, 11.4, 9.7, 41.8),    # The dropoff
                          (0.0, 5.5, 5.4, 60.1),     # Faculty offices
                          (5.4, 58.5, 32.5, 60.1)]   # Biology territory

        self.validPosList = []
        self.olinMap = None
        self.currentMap = None
        self.getOlinMap()




    def initializeParticles(self, partNum):
        for i in range(partNum):
            self.addRandomParticle()



    def addRandomParticle(self):
        """generating a new list with random possibility all over the map"""
        print "adding random particles"
        posAngle = np.random.uniform(0, 2*np.pi)  # radians :P       WHY NOT JUST 0 TO 2Pi?
        posX = np.random.uniform(0, self.worldX)
        posY = np.random.uniform(0, self.worldY)
        if self.isValid((posX,posY)):
            self.validPosList.append((posX, posY, posAngle))
        else:
            self.addRandomParticle()


    def addNearbyParticle(self):
        print "adding nearby particles"
        addList = []
        for part in self.validPosList:
            posX = part[0]
            posY = part[1]
            posAngle = part[2]
            newAngle = np.random.uniform(posAngle - np.pi/2, posAngle + np.pi/2)
            newX = np.random.uniform(posX - 5, posX + 5)
            newY = np.random.uniform(posY - 5, posY + 5)
            newX = int(newX)
            newY = int(newY)

            if self.isValid((newX, newY)):
                addList.append((newX, newY, newAngle))
        self.validPosList.extend(addList)



    def particleMove(self, moveDist, moveAngle):
        """updating the information of the points when the robot moves"""
        print "in particleMove"
        moveList = []
        for i in range(len(self.validPosList)):
            posPoint = self.validPosList[i]
            posAngle = posPoint[2] + moveAngle
            if posAngle>2*np.pi:
                posAngle -= 2*np.pi
            print "new angle ", np.degrees(posAngle)

            posX = posPoint[0] + moveDist * math.cos(posAngle)
            posY = posPoint[1] + moveDist * math.sin(posAngle)
            print "old loc: ", posPoint[0], posPoint[1]
            print "new loc: ", posX, posY

            moveList.append((posX, posY, posAngle))

        self.validPosList = moveList
        print "length ", len(self.validPosList)
        self.update(self.validPosList)



    def update(self, posList):
        """Updating the possibility list and removing the not valid ones"""
        print "in update ", "length ", len(self.validPosList)
        updatedList = []

        #removing any nodes that are now invalid and adding particles nearby the ones that remain
        for particle in posList:
            print "removing invalid nodes"
            if self.isValid(particle):
                updatedList.append(particle)
                print "updated list ", len(updatedList)

        self.validPosList = updatedList
        print "length ", len(self.validPosList)
        #self.addNearbyParticle()
        # Adding randomized particles to the map
        # for i in range(4):
        #     self.addRandomParticle()
        print "length ", len(self.validPosList)




    def isValid(self, posPoint):
        """checking if the particle is within the obstacle areas
        return false if it's no longer a valid point"""
        posX = posPoint[0]
        posY = posPoint[1]

        for rect in self.obstacles:

            if (posX >= rect[0] and posX <= rect[2]) and (posY >= rect[1] and posY <= rect[3]):
                print "out of bounds"
                return False

        return True



    def getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(self.mapFile, 20)
        map2 = np.flipud(origMap)
        orientMap = np.rot90(map2)
        self.olinMap = orientMap.copy()
        (self.mapHgt, self.mapWid, self.mapDep) = self.olinMap.shape
        self.drawObstacles()
        self.currentMap = self.olinMap.copy()


    def drawObstacles(self):
        """Draws the obstacles on the currently passed image
        NOTE: the obstacle positions must be converted."""
        for obst in self.obstacles:
            (lrX, lrY, ulX, ulY) = obst
            mapUL = self.convertWorldToMap(ulX, ulY)
            mapLR = self.convertWorldToMap(lrX, lrY)
            cv2.rectangle(self.olinMap, mapUL, mapLR, (255, 0, 0), thickness=2)


    def drawParticles(self, color):
        # self.currentMap = self.olinMap.copy()        # Ultimately we want this line, but for debugging
        for part in self.validPosList:
            (x, y, head) = part
            self.drawSingleParticle(self.currentMap, x, y, head, color)
        cv2.imshow("Particles", self.currentMap  )
        cv2.waitKey(20)


    def drawSingleParticle(self, image, wldX, wldY, heading, color):
        pointLen = 1.0  # meters
        pointX = wldX + (pointLen * math.cos(heading))
        pointY = wldY + (pointLen * math.sin(heading))
        print "WLD Center:", (wldX, wldY), "   Point:", (pointX, pointY), "Heading: ", np.degrees(heading)


        mapCenter = self.convertWorldToMap(wldX, wldX)
        mapPoint = self.convertWorldToMap(pointX, pointY)
        print "MAP Center:", mapCenter, "   Point:", mapPoint
        cv2.circle(image, mapCenter, 6, color, -1)
        cv2.line(image, mapCenter, mapPoint, color)



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
        mapY = self.mapHgt - 1 - pixelX
        return int(mapX), int(mapY)




test = monteCarloLoc()
test.initializeParticles(5)

list = test.getParticles()
test.drawParticles((255,100,0))
cv2.waitKey(0)

for i in range(20):
    print "in the for loop"
    test.particleMove(1.0,0)
    test.drawParticles((0,0,255-i*12))
    cv2.waitKey(0)

cv2.destroyAllWindows()

