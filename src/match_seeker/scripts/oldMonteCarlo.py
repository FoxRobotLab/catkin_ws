#!/usr/bin/env python


'''This class is used to implement Monte Carlo Localization'''

import numpy as np
import cv2
from DataPaths import basePath
from markLocations import readMap
import math


class monteCarloLoc():
    def __init__(self):
        self.worldX = 40.75  # meters
        self.worldY = 60.1  # meters
        self.mapWid = 1203  # pixels
        self.mapHgt = 816  # pixels
        self.maxLen = 500  # num of particles
        # self.worldX, self.worldY = self.convertMapToWorld(self.mapWid - 1, self.mapHgt - 1)
        # print self.worldX, self.worldY
        # x1, y1, x2, y2 (x1 <= x2; y1 <= y2)
        # bounding tuples of rectangles that represent classrooms, stairs, and walls
        self.obstacles = [(32.5, 0.0, 50.4, 60.1),  # top obstacle across classrooms
                          (23.7, 0.0, 32.5, 5.5),  # ES office more or less
                          (0.0, 0.0, 17.5, 5.5),  # Near the origin
                          (21.0, 7.2, 30.0, 42.2),  # Covers 205 and 250
                          (9.7, 7.2, 18.9, 42.4),  # Covers 258 etc
                          (7.0, 11.4, 9.7, 41.8),  # The dropoff
                          (0.0, 5.5, 5.4, 60.1),  # Faculty offices
                          (5.4, 58.5, 32.5, 60.1),  # Biology territory
                          (17.5, 0, 21.7, 5.5),  # long wall of robot lab
                          (22.8, 0, 23.7, 5.5)]  # short wall of robot lab

        self.validPosList = []
        self.weightedList = []

        # Map stuff
        self.olinMap = None
        self.currentMap = None
        self.getOlinMap(basePath + "res/map/olinNewMap.txt")

        # random perturbation constants: std dev for movement and turning as a percentage of movement
        # so if movement in x direction is 500 cm, then the std dev is 100
        self.sigma_fwd_pct = 0.2
        self.sigma_theta_pct = 0.05

    def getOlinMap(self, mapFile):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(mapFile, 20)
        map2 = np.flipud(origMap)
        orientMap = np.rot90(map2)
        self.olinMap = orientMap.copy()
        (self.mapHgt, self.mapWid, self.mapDep) = self.olinMap.shape
        self.drawObstacles()
        self.currentMap = self.olinMap.copy()

    def initializeParticles(self, partNum):
        self.maxLen = partNum
        # print "initializing particles"
        for i in range(partNum):
            self.validPosList.append(self.addRandomParticle())

    def addRandomParticle(self):
        """generating a new list with random possibility all over the map"""
        # print "adding random particles"
        posAngle = np.random.uniform(0, 2 * np.pi)  # radians :P
        posX = np.random.uniform(0, self.worldX)
        posY = np.random.uniform(0, self.worldY)
        if self.isValid((posX, posY)):
            return (posX, posY, posAngle)
        else:
            return self.addRandomParticle()

    def addNearbyParticle(self, particles):
        # print "adding nearby particles"
        addList = []
        for part in particles:
            (posX, posY, posAngle) = part
            while True:
                deltaTh = 2 * self.sigma_theta_pct * np.random.normal()
                deltaX = 5 * self.sigma_fwd_pct * np.random.normal()
                deltaY = 5 * self.sigma_fwd_pct * np.random.normal()

                newAngle = posAngle + deltaTh
                newX = posX + deltaX
                newY = posY + deltaY
                # newAngle = np.random.uniform(posAngle - np.pi/2, posAngle + np.pi/2)
                # newX = np.random.uniform(posX - 1.0, posX + 1.0)
                # newY = np.random.uniform(posY - 1.0, posY + 1.0)

                if self.isValid((newX, newY)):
                    addList.append((newX, newY, newAngle))
                    break
        return addList

    def particleMove(self, moveInfo):
        """updating the information of the points when the robot moves"""
        # print "in particleMove"

        moveX, moveY, moveAngle = moveInfo
        moveAngle = np.radians(moveAngle)
        moveList = []
        for i in range(len(self.validPosList)):
            posPoint = self.validPosList[i]
            posAngle = posPoint[2] + moveAngle
            if posAngle > 2 * np.pi:
                posAngle -= 2 * np.pi

            posX = posPoint[0] + moveX
            posY = posPoint[1] + moveY

            if (self.isValid((posX, posY, posAngle))):
                moveList.append((posX, posY, posAngle))
                # else:
                #     newPoint = self.addNormParticles(posPoint[0], posPoint[1], posPoint[2])
                #     moveList.append(newPoint)

        # print "Moving", moveDist, moveAngle
        self.validPosList = moveList

        # print "Moving", len(self.validPosList)
        # self.update(self.validPosList)

    def addNormParticles(self, posX, posY, posAngle):

        # print "Adding normal distribution particles"

        newAngle = np.random.normal(posAngle, np.pi / 4)
        newX = np.random.normal(posX, 1.0)
        newY = np.random.normal(posY, 1.0)

        if self.isValid((newX, newY, newAngle)):
            return (newX, newY, newAngle)
            # print newX, newY, newAngle
        else:
            return self.addNormParticles(posX, posY, posAngle)

    # Add random particle part is not up-to-date.
    # But we are not calling update.
    def update(self, posList):
        """Updating the possibility list and removing the not valid ones"""
        # print "in update ", "length ", len(self.validPosList)
        updatedList = []

        # removing any nodes that are now invalid and adding particles nearby the ones that remain
        for particle in posList:
            # print "removing invalid nodes"
            if self.isValid(particle):
                updatedList.append(particle)
                # print "updated list ", len(updatedList)
        self.validPosList = updatedList

        addList = self.addNearbyParticle(self.validPosList)  # add one point near every valid particle

        self.validPosList.extend(addList)
        for i in range(2):
            self.addRandomParticle()  # add a few particles in random locations
            # print "length at update ", len(self.validPosList)

    def isValid(self, posPoint):
        """checking if the particle is within the obstacle areas
        return false if it's no longer a valid point"""
        posX = posPoint[0]
        posY = posPoint[1]

        for rect in self.obstacles:
            if (posX >= rect[0] and posX <= rect[2]) and (posY >= rect[1] and posY <= rect[3]):
                # print "out of bounds"
                return False
        return True

    def mclCycle(self, matchLocs, matchScores, odometry, odomScore, moveInfo):
        """ Takes in important Localizer information and calls all relevant methods in the MCL"""
        self.particleMove(moveInfo)
        self.calcWeights(matchLocs, matchScores, odometry, odomScore)
        self.validPosList = self.getSample()
        self.calcWeights(matchLocs, matchScores, odometry, odomScore)
        # self.update(self.validPosList)

    def calcWeights(self, matchLocs, matchScores, odometry, odomScore):
        """ Weight for each particle based on if it is a possible location given the Localizer data. """
        ### our faked calculations based on proximity to a hardcoded keypoint
        # keyX = 7.0
        # keyY = 7.0
        # weights = []
        # for i in range(len(self.validPosList)):
        #     x = self.validPosList[i][0]
        #     y = self.validPosList[i][1]
        #     dist = self._euclidDist((keyX, keyY), (x, y))
        #     wht = 60 - dist
        #     weights.append(wht)
        # arrayWeights = np.array(weights, np.float64)
        # self.normedWeights = arrayWeights / arrayWeights.sum()
        #
        # for i in range(len(self.validPosList)):
        #     part = self.validPosList[i]
        #     self.weightedList.append((normedWeights[i],part[0],part[1],part[2]))

        weights = []
        for i in range(len(self.validPosList)):
            # for each particle, look at its distance from the odomLoc & each matchLoc scaled by the location's certainty score
            x = self.validPosList[i][0]
            y = self.validPosList[i][1]
            minDist = self._euclidDist((odometry[0], odometry[1]), (x, y))
            minScore = odomScore

            for m in range(len(matchLocs)):
                matchDist = self._euclidDist((matchLocs[m][0], matchLocs[m][1]), (x, y))
                if minDist > matchDist:
                    minDist = matchDist
                    minScore = matchScores[m]

            weights.append((75 - minDist) * (minScore / 100))  # append the maximum weight for each particle
            # print "x and y: ", x, " ", y, "min dist index: ", str(minIndex)
        arrayWeights = np.array(weights, np.float64)
        self.normedWeights = arrayWeights / arrayWeights.sum()  # normalize the weights for all particles
        print "arrayWeights sum:", arrayWeights.sum()
        if arrayWeights.sum() == 0:
            print "arryaWeights sum is zero."

    def probableLocation(self):
        index = np.where(self.normedWeights == max(self.normedWeights))
        index = index[0].take(0)
        # print "index", index, "type", type(index)
        point = self.validPosList[index]
        # print "Prob loc, length: ", len(self.validPosList)
        # print "point type", type(point[0])
        return point

    def getSample(self):
        # weights = [ x[0] for x in self.weightedList]
        list_idx_choices = np.random.multinomial(self.maxLen, self.normedWeights)
        # list_idx_choices of form [0, 0, 2, 0, 1, 0, 4] for length 7 list_to_sample
        # print list_idx_choices
        total_particles = 0
        sampleList = []
        for idx, count in enumerate(list_idx_choices):
            currParticle = self.validPosList[idx]
            while count > 0:
                total_particles += 1
                if count == 1:
                    sampleList.append(currParticle)
                else:
                    newParticle = self.addNormParticles(currParticle[0], currParticle[1], currParticle[2])
                    sampleList.append(newParticle)
                # elif count < 3 or total_particles < self.maxLen - 10:  # Stop duplicating if max reached
                #     # Need to add copies to new list, not just identical references!
                #     new_particle_list = self.addNearbyParticle([self.validPosList[idx]])
                #     sampleList.extend(new_particle_list)
                # elif count < 3 or total_particles < self.maxLen:
                #     new_particle = self.addRandomParticle()
                #     sampleList.append(new_particle)
                count -= 1
        return sampleList

    def centerOfMass(self):

        """Calculating the center of mass of the position cluster in order to
        give a prediction about where the robot is.
        Returns the possible location of the robot"""

        # print "Center of Mass, length: ", len(self.validPosList)

        xList = [pos[0] for pos in self.validPosList]
        yList = [pos[1] for pos in self.validPosList]
        angleList = [pos[2] for pos in self.validPosList]

        cgx = np.sum(xList * self.normedWeights) / np.sum(self.normedWeights)
        cgy = np.sum(yList * self.normedWeights) / np.sum(self.normedWeights)
        cgAngle = np.sum(angleList * self.normedWeights) / np.sum(self.normedWeights)

        cgx = cgx.take(0)
        cgy = cgy.take(0)
        cgAngle = cgAngle.take(0)
        # print "cgx", cgx, "cgy", cgy, "cgangle", cgAngle
        return cgx, cgy, cgAngle

    def _euclidDist(self, (x1, y1), (x2, y2)):
        """Given two tuples containing two (x, y) points, this computes the straight-line distsance
        between the two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def drawObstacles(self):
        """Draws the obstacles on the currently passed image
        NOTE: the obstacle positions must be converted."""
        for obst in self.obstacles:
            (lrX, lrY, ulX, ulY) = obst
            mapUL = self.convertWorldToMap(ulX, ulY)
            mapLR = self.convertWorldToMap(lrX, lrY)
            cv2.rectangle(self.olinMap, mapUL, mapLR, (255, 0, 0), thickness=2)

    def drawParticles(self, color):
        self.currentMap = self.olinMap.copy()  # Ultimately we want this line, but for debugging

        maxWeight = max(self.normedWeights)

        for i in range(len(self.validPosList)):
            point = self.validPosList[i]
            weight = self.normedWeights[i] / maxWeight
            (x, y, head) = point
            b, g, r = color
            newColor = (b * weight, g * weight, r * weight)
            self.drawSingleParticle(self.currentMap, x, y, head, newColor)
        cv2.imshow("Particles", self.currentMap)
        cv2.waitKey(20)

    def drawSingleParticle(self, image, wldX, wldY, heading, color):
        pointLen = 0.5  # meters
        pointX = wldX + (pointLen * math.cos(heading))
        pointY = wldY + (pointLen * math.sin(heading))
        # print "WLD Center:", (wldX, wldY), "   Point:", (pointX, pointY), "Heading: ", np.degrees(heading)


        mapCenter = self.convertWorldToMap(wldX, wldY)
        mapPoint = self.convertWorldToMap(pointX, pointY)
        # print "MAP Center:", mapCenter, "   Point:", mapPoint
        cv2.circle(image, mapCenter, 4, color, -1)
        cv2.line(image, mapCenter, mapPoint, color)

    def drawBlank(self):
        width = self.mapWid
        blank_image = np.zeros((width, width, 3), np.uint8)
        return blank_image

    def getParticles(self):
        return self.validPosList

    # Not working properly.
    def convertMapToWorld(self, mapX, mapY):
        """Converts coordinates in pixels, on the map, to coordinates (real-valued) in
        meters. Note that this also has to adjust for the rotation and flipping of the map."""
        # First flip x and y values around...
        flipY = self.mapWid - 1 - mapX
        flipX = self.mapHgt - 1 - mapY

        # flipY = mapX + self.mapWid - 1
        # flipX = mapY + self.mapHgt - 1
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



if __name__ == '__main__':
    test = monteCarloLoc()
    test.initializeParticles(1000)
    # print "total len ", len(test.validPosList)
    test.drawParticles((255, 100, 0))
    cv2.waitKey(0)

    for i in range(50):
        # print "in the for loop", i

        test.calcWeights(0, 0, 0, 0)
        list = test.getSample()

        # print "total len ", len(test.validPosList)
        # print "sample len ", len(list)
        test.validPosList = list

        # test.particleMove(1.0,0)
        # print test.validPosList[0]
        # test.addNormParticles(test.validPosList[0][0], test.validPosList[0][1], test.validPosList[0][2])
        test.calcWeights(0, 0, 0, 0)
        x, y, head = test.centerOfMass()
        print "Center of mass", x, y, head
        test.drawParticles((0, 0, 255 - i * 12))
        test.drawSingleParticle(test.currentMap, x, y, head, (0, 255, 0))

        index = np.where(test.normedWeights == max(test.normedWeights))
        probX, probY, probAngle = test.validPosList[index[0]]
        print "most likely loc", probX, probY, probAngle
        test.drawSingleParticle(test.currentMap, probX, probY, probAngle, (255, 255, 0))

        cv2.imshow("Particles", test.currentMap)
        cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

