#!/usr/bin/env python


'''This class is used to implement Monte Carlo Localization'''


import numpy as np
import cv2
import math

from Particle import Particle
from OlinWorldMap import WorldMap
# import matplotlib.pyplot as plt


class monteCarloLoc(object):

    def __init__(self, olinMap):
        self.olinMap = olinMap
        self.maxLen = 500   # num of particles
        self.maxWeight = 0.0
        self.sumWeight = 0.9
        self.centerParticle = None

        # bounding tuples of rectangles that represent classrooms, stairs, and walls
        #  self.obstacles = [(32.5, 0.0, 50.4,  60.1),   # top obstacle across classrooms
        #                    (23.7, 0.0, 32.5, 5.5),    # ES office more or less
        #                    (0.0, 0.0, 17.5, 5.5),     # Near the origin
        #                    (21.0, 7.2, 30.0, 42.2),   # Covers 205 and 250
        #                    (9.7, 7.2, 18.9, 42.4),    # Covers 258 etc
        #                    (7.0, 11.4, 9.7, 41.8),    # The dropoff
        #                    (0.0, 5.5, 5.4, 60.1),     # Faculty offices
        #                    (5.4, 58.5, 32.5, 60.1),   # Biology territory
        #                    (17.5, 0, 21.7, 5.5),       # long wall of robot lab
        #                    (22.8, 0, 23.7, 5.5)]      # short wall of robot lab
        #

        self.validPosList = []
        self.currentData = {}


        # random perturbation constants: std dev for movement and turning as a percentage of movement
        # so if movement in x direction is 500 cm, then the std dev is 100
        # self.sigma_fwd_pct = 0.2
        # self.sigma_theta_pct = 0.05

        # plt
        # plt.ion()
        # plt.show()


    def initializeParticles(self, partNum,point):
        """Given partNum, the number of particles, this generates that many
        randomly placed particles."""
        self.maxLen = partNum
        # print "initializing particles"
        for i in range(partNum):
            newParticle = Particle(self.olinMap, mode='spec',initPose=point)
            self.validPosList.append(newParticle)


    def mclCycle(self, mclData, moveInfo, windowName = "MCL Display"):
        """ Takes in important Localizer information and calls all relevant methods in the MCL"""
        self.currentData = mclData


        self.particleMove(moveInfo)
        # print "after particle move", len(self.validPosList)
        matchParticles = self.seedNodesAtMatches()
        matchCopies = self.seedNodesAtMatches()
        self.validPosList.extend(matchParticles)

        self.calcWeights()
        self.normalizeWeights()
        self.validPosList.sort(key = lambda p: p.weight)
        self.resampleParticles()
        # self.normalizeWeights()
        self.calcWeights()
        self.normalizeWeights()
        self.centerOfMass()
        var = self.calculateVariance()
        if self.currentData["odomScore"] < 1 and var < 3.0:
            self.scatter()

        self.olinMap.cleanMapImage(obstacles=True,cells=True,drawCellNum=True)
        self.drawParticles(self.validPosList, (0, 0, 0), fill = False)      # draw set of particles  in black
        self.drawParticles(matchCopies[1:], (255, 0, 255))   # draw particles for matched images in magenta
        self.drawParticles(matchCopies[:1], (0, 170, 255))   # draw particle for odometry location in orange
        self.drawParticles([self.centerParticle], (0, 255, 0))  # draw particle for center of mass in green

        self.olinMap.displayMap(windowName)

        return self.centerParticle.getLoc(), var


    def calculateVariance(self):

        if self.centerParticle == None or not self.centerParticle.isValid():
            return 300.0


        centerX, centerY, centerAngle = self.centerParticle.getLoc()
        vx = 0
        vy = 0

        for particle in self.validPosList:
            x, y, angle = particle.getLoc()
            weight = particle.getWeight()
            vx += weight * (x - centerX)**2
            vy += weight * (y - centerY)**2

        variance = vx+vy
        return variance


    def particleMove(self, moveInfo):
        """updating the information of the points when the robot moves"""
        # print "in particleMove"
        moveX, moveY, moveAngle = moveInfo
        moveList = []
        for posPoint in self.validPosList:
            posPoint.moveParticle(moveX, moveY, moveAngle)
            if posPoint.isValid():
                moveList.append(posPoint)
            # else:
            #    print "NOT VALID"
        self.validPosList = moveList


    def seedNodesAtMatches(self):
        """Given the odometry location, odomLoc, and the current match locations, matchLocs, generate
        a set of particles at those locations, two for each matched-image location."""
        particles = [Particle(self.olinMap, self.currentData['odomPose'])]
        for loc in self.currentData['matchPoses']:
            particles.append(Particle(self.olinMap, loc))
            particles.append(Particle(self.olinMap, loc))
        return particles



    def calcWeights(self):
        """ Weight for each particle based on if it is a possible location given the Localizer data.
        Inputs:
        matchLocs: the list of (3) locations for the best matching images
        matchScores: the scores those images got
        odometry: the pose from the odometry data
        odomScore: the confidence score for odometry: degrades over time"""
        for posPoint in self.validPosList:
            # for each particle, look at its distance from the odomLoc & each matchLoc scaled by the location's certainty score
            posPoint.calculateWeight(self.currentData)

        weights = self.weightList()
        self.maxWeight = max(weights)
        self.sumWeight = sum(weights)
        assert self.sumWeight != 0


    def normalizeWeights(self):
        """Assuming weights have been set, normalize them so that they add up to 1.0."""
        for particle in self.validPosList:
            particle.normWeight(self.sumWeight)

    def scatter(self, loc = None):
        """sends the x, y, of the center of mass to the particle class for every particle.
        this generates a new particle in a uniform area around the center of mass with a random heading."""
        if loc == None:
            particle = self.centerParticle
            x, y, heading = particle.getLoc()
        else:
            x, y, heading = loc

        for part in self.validPosList:
            part.scatter(x, y, 5.0)

    def resampleParticles(self):
        """Generate the next set of particles by resampling based on their current weights, which should
        sum to one as a probability distribution. If generating more than one copy of a current particle,
        all after the first are perturbed copies."""

        # multinomial generates a list as long as weights (which is the same as the number of particles).
        # in this list at each position is how many copies of that particle should be sampled
        weights = self.weightList()
        list_idx_choices = np.random.multinomial(self.maxLen, weights)

        # plt
        # plt.clf()
        # plt.plot(weights)
        # plt.draw()
        # plt.pause(0.001)

        # generate particles for new particle list
        total_particles = 0
        sampleList = []
        for idx, count in enumerate(list_idx_choices):
            # print idx, count
            currParticle = self.validPosList[idx]
            while count > 0:
                total_particles += 1
                if count == 1:
                    sampleList.append(currParticle)
                    # print "currParticle", currParticle.heading
                else:
                    newParticle = currParticle.makePerturbedCopy()
                    # newParticle.calculateWeight(self.currentData)
                    sampleList.append(newParticle)
                    # print "newParticle heading", newParticle.heading
                count -= 1
        self.validPosList = sampleList


    def centerOfMass(self):
        """Calculating the center of mass of the position cluster in order to
        give a prediction about where the robot is.
        Returns the possible location of the robot"""

        # print "Center of Mass, length: ", len(self.validPosList)

        weightedSumX = 0
        weightedSumY = 0
        weightedSumAngle = 0
        weightList = []
        angleList = []

        for particle in self.validPosList:
            weightedSumX += particle.getScaledX()
            weightedSumY += particle.getScaledY()
            weightList.append(particle.getWeight())
            angleList.append(particle.getLoc()[2])

        totalWeight = sum(weightList)


        cgx = weightedSumX / totalWeight
        cgy = weightedSumY / totalWeight
        cgAngle = self.circular_mean(weightList, angleList)

        cgParticle = Particle(self.olinMap, (cgx, cgy, cgAngle))
        # print "cgx", cgx, "cgy", cgy, "cgangle", cgAngle
        self.centerParticle = cgParticle


    def circular_mean(self, weights, angles):
        """The helper function to calculate weighted average of angles."""
        x = y = 0.
        for angle, weight in zip(angles, weights):
            x += math.cos(math.radians(angle)) * weight
            y += math.sin(math.radians(angle)) * weight

        mean = math.degrees(math.atan2(y, x))
        return mean


    def drawParticles(self, particleList, color = (0, 0, 0), size = 4, fill = True, shading = False):
        for point in particleList:
            if shading:
                weight = point.getWeight() * self.maxWeight
                b, g, r = color
                color = (b * weight, g * weight, r * weight)
            self.olinMap.drawPose(point, size, color, fill)

    def weightList(self):
        """Builds and returns a list of the weights of the current valid particle list."""
        return [pose.getWeight() for pose in self.validPosList]



if __name__ == '__main__':
    olinMap = WorldMap()

    p = Particle(olinMap, (31, 56, 270))
    print(p)
    olinMap.drawPose(p)
    olinMap.displayMap()
    cv2.waitKey()

    while True:
        olinMap.cleanMapImage(obstacles=True,cells=True, drawCellNum=True)
        dir = input("change in heading: ")
        dir = int(dir)
        p.moveParticle(0, 0, dir)
        print(p)
        olinMap.drawPose(p)
        olinMap.displayMap()
        cv2.waitKey(10)

    # test = monteCarloLoc(olinMap)
    # test.initializeParticles(1,(31,56,270))
    # # print "total len ", len(test.validPosList)
    # test.drawParticles(test.validPosList, (255,100,0))
    # cv2.imshow("Particles", olinMap.currentMapImg)
    # cv2.waitKey(0)
    # test.particleMove((1, 0, 0))
    # test.drawParticles(test.validPosList, (255, 100, 0))


    # for i in range(50):
    #     # print "in the for loop", i
    #
    #     test.calcWeights(0, 0, 0, 0)
    #     list = test.getSample()
    #
    #     # print "total len ", len(test.validPosList)
    #     # print "sample len ", len(list)
    #     test.validPosList = list
    #
    #     # test.particleMove(1.0,0)
    #     # print test.validPosList[0]
    #     # test.addNormParticles(test.validPosList[0][0], test.validPosList[0][1], test.validPosList[0][2])
    #     test.calcWeights(0, 0, 0, 0)
    #     x, y, head = test.centerOfMass()
    #     print "Center of mass", x, y, head
    #     test.drawParticles(test.validPosList, (0,0,255-i*12))
    #     test.drawSingleParticle(test.currentMap, x, y, head, (0, 255, 0))
    #
    #
    #     index = np.where(test.normedWeights == max(test.normedWeights))
    #     probX, probY, probAngle = test.validPosList[index[0]]
    #     print "most likely loc", probX, probY, probAngle
    #     test.drawSingleParticle(test.currentMap, probX, probY, probAngle, (255, 255, 0))
    #
    #
    # cv2.imshow("Particles", olinMap.currentMapImg)
    # cv2.waitKey(0)
    # #
    #
    #
    # cv2.destroyAllWindows()
    #
    #
    # test = monteCarloLoc(olinMap)
    # test.initializeParticles(10, (15, 6, 180))
    # mclDataFake = {'matchPoses': [(12.8, 6.3, 180), (10.0, 6.1, 180), (13.1, 6.5, 0)],
    #                'matchScores': [67.0, 55.2, 41.3],
    #                'odomPose': (12.4, 6.45, 169),
    #                'odomScore': 89.0}
    # centerPos = test.mclCycle(mclDataFake, (0.24, 0.003, 2.0))
    # print centerPos
    # cv2.waitKey()


    #
    # part = test.validPosList[0]
    # part.setLoc(15.0,50.0,112)
    # test.drawParticles(test.validPosList, (0,0,255))
    #
    # for i in range(50):
    #     test.particleMove((-1.0,-1.0,0.0))
    #     print test.validPosList[0]
    #     test.drawParticles(test.validPosList, (0,0,255))
    #     if len(test.validPosList) == 0:
    #         break
    #     test.olinMap.displayMap()
    #     cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()


    # test.maxLen = 10

    # for i in range(len(test.validPosList)):
    #     particle = test.validPosList[i]
    #     particle.setWeight(float(i))
    #
    #
    # weights = [p.getWeight() for p in test.validPosList]
    # print "Weights before normalized", weights
    # sumWeight = sum(weights)
    #
    # for particle in test.validPosList:
    #     particle.normWeight(sumWeight)
    #
    # weights = [p.getWeight() for p in test.validPosList]
    # print "Weights after normalized", weights
    #
    # sampleList = test.getSample()
    #
    # print "Here"
    #
    # str = raw_input("Say something")
    # # plt.show()





    # for particle in sampleList:
    #     print particle
