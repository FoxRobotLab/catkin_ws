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
        print self.worldX, self.worldY
        self.mapFile =  basePath + "res/map/olinNewMap.txt"

        # x1, y1, x2, y2 (x1 <= x2; y1 <= y2)
         # bounding tuples of rectangles that represent classrooms, stairs, and walls
        self.obstacles = [(50.4, 60.1, 32.5, 0.0), (32.5, 5.5, 23.7, 0.0),
                          (17.5, 5.5, 0.0, 0.0), (30.0, 42.2, 21.0, 7.2),
                          (18.9, 42.4, 9.7, 7.2), (9.7, 41.8, 7.0, 11.4),
                          (5.4, 60.1, 0.0, 5.5), (32.5, 60.1, 5.4, 58.5)]

        self.validPosList = []
        for i in range(5):
            self.addRandomParticle()

    def addRandomParticle(self):
        """generating a new list with random possibility all over the map"""
        print "adding random particles"
        posAngle = np.random.uniform(-2*np.pi, 2*np.pi)  # radians :P
        # posAngle = math.radians(0) + math.pi/2
        # posX = 22.2
        # posY = 6.5
        # posX, posY = np.random.uniform(0, self.mapWid, 2)
        posX = np.random.uniform(0, self.worldX)
        posY = np.random.uniform(0, self.worldY)
        posX = int(posX)
        posY= int(posY)
        # self.validPosList.append((posX, posY, posAngle))
        if self.isValid((posX,posY)):
            self.validPosList.append((posX, posY, posAngle))
        # else:
        #     self.addRandomParticle()


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

        line = 20
        newX = int(x + (line * math.cos(heading)))
        newY = int(y + (line * math.sin(heading)))

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
        mapY = self.mapHgt - 1 - pixelX
        return int(mapX), int(mapY)

    def drawParticles(self,color):
        for i in range(len(self.validPosList)):
            (x, y, head) = self.validPosList[i]
            drawX = int(x)
            drawY = int(y)
            test.drawPosition(map, drawX, drawY, head, color)

test = monteCarloLoc()
test.getOlinMap()
map = test.olinMap.copy()
# blank = test.drawBlank()
list = test.getParticles()
# blank[194:194+map.shape[0], 0:map.shape[1]] = map
for rect in test.obstacles:
    cv2.rectangle(map,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),thickness=2)

test.drawParticles((255,100,0))
cv2.imshow("points", map)
cv2.waitKey(0)

# im = cv2.resize(blank,(900,900), interpolation = cv2.INTER_AREA)
# cv2.imshow("drawing", map)

for i in range(20):
    print "in the for loop"
    test.particleMove(20,0)
    test.drawParticles((0,0,255-i*20))
    cv2.imshow("points", map)
    cv2.waitKey(0)

cv2.destroyAllWindows()

