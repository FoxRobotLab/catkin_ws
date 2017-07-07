#!/usr/bin/env python


'''This class is used to implement Monte Carlo Localization'''


import numpy as np
import cv2
from DataPaths import basePath
from Particle import Particle
import math

# import matplotlib.pyplot as plt


class monteCarloLoc(object):

    def __init__(self, olinMap):
        self.olinMap = olinMap
        self.maxLen = 500   # num of particles
        self.maxWeight = 0.0
        self.sumWeight = 0.9

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
        self.weightedList = []


        # random perturbation constants: std dev for movement and turning as a percentage of movement
        # so if movement in x direction is 500 cm, then the std dev is 100
        # self.sigma_fwd_pct = 0.2
        # self.sigma_theta_pct = 0.05

        # plt
        # plt.ion()
        # plt.show()


    def initializeParticles(self, partNum):
        """Given partNum, the number of particles, this generates that many
        randomly placed particles."""
        self.maxLen = partNum
        # print "initializing particles"
        for i in range(partNum):
            newParticle = Particle(self.olinMap, mode='random')
            self.validPosList.append(newParticle)


    def mclCycle(self, matchLocs, matchScores, odometry, odomScore, moveInfo):
        """ Takes in important Localizer information and calls all relevant methods in the MCL"""
        self.particleMove(moveInfo)
        # print "after particle move", len(self.validPosList)
        matchParticles = self.seedNodesAtMatches(matchLocs,odometry)
        self.validPosList.extend(matchParticles)
        self.calcWeights(matchLocs, matchScores, odometry, odomScore)
        self.normalizeWeights()
        self.validPosList.sort(key = lambda p: p.weight)
        self.resampleParticles(matchLocs, matchScores, odometry, odomScore)
        self.normalizeWeights()
        # self.calcWeights(matchLocs, matchScores, odometry, odomScore)

        self.olinMap.cleanMapImage(obstacles=True)
        self.drawParticles(self.validPosList, (0,0,255))      # draw set of particles  in red
        self.drawParticles(matchParticles[1:], (255,0,255))   # draw particles for matched images in magenta
        self.drawParticles(matchParticles[:1], (255, 0, 0))   # draw particle for odometry location in blue
        self.olinMap.displayMap()



    def particleMove(self, moveInfo):
        """updating the information of the points when the robot moves"""
        # print "in particleMove"
        moveX, moveY, moveAngle = moveInfo
        moveList = []
        for posPoint in self.validPosList:
            posPoint.moveParticle(moveX, moveY, moveAngle)
            if posPoint.isValid():
                moveList.append(posPoint)
        self.validPosList = moveList


    def seedNodesAtMatches(self, matchLocs, odomLoc):
        """Given the odometry location, odomLoc, and the current match locations, matchLocs, generate
        a set of particles at those locations, two for each matched-image location."""
        particles = [Particle(self.olinMap, odomLoc)]
        for loc in matchLocs:
            particles.append(Particle(self.olinMap, loc))
            particles.append(Particle(self.olinMap, loc))
        return particles



    def calcWeights(self, matchLocs, matchScores, odometry, odomScore):
        """ Weight for each particle based on if it is a possible location given the Localizer data.
        Inputs:
        matchLocs: the list of (3) locations for the best matching images
        matchScores: the scores those images got
        odometry: the pose from the odometry data
        odomScore: the confidence score for odometry: degrades over time"""
        for posPoint in self.validPosList:
            # for each particle, look at its distance from the odomLoc & each matchLoc scaled by the location's certainty score
            posPoint.calculateWeight(matchLocs, matchScores, odometry, odomScore)

        weights = self.weightList()
        self.maxWeight = max(weights)
        self.sumWeight = sum(weights)
        assert self.sumWeight != 0


    def normalizeWeights(self):
        """Assuming weights have been set, normalize them so that they add up to 1.0."""
        for particle in self.validPosList:
            particle.normWeight(self.sumWeight)


    def resampleParticles(self, matchLocs, matchScores, odomLoc, odomScore):
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
            print idx, count
            currParticle = self.validPosList[idx]
            while count > 0:
                total_particles += 1
                if count == 1:
                    sampleList.append(currParticle)
                else:
                    newParticle = currParticle.makePerturbedCopy()
                    newParticle.calculateWeight(matchLocs, matchScores, odomPose, odomScore)
                    sampleList.append(newParticle)
                count -= 1
        return sampleList


    def centerOfMass(self):
        """Calculating the center of mass of the position cluster in order to
        give a prediction about where the robot is.
        Returns the possible location of the robot"""

        # print "Center of Mass, length: ", len(self.validPosList)

        weightedSumX = 0
        weightedSumY = 0
        weightedSumAngle = 0
        totalWeight = 0

        for particle in self.validPosList:
            weightedSumX += particle.getScaledX()
            weightedSumY += particle.getScaledY()
            weightedSumAngle += particle.getScaledAngle()
            totalWeight += particle.getWeight()

        cgx = weightedSumX/totalWeight
        cgy = weightedSumY/totalWeight
        cgAngle = weightedSumAngle/totalWeight

        cgParticle = Particle(cgx, cgy, cgAngle)
        # print "cgx", cgx, "cgy", cgy, "cgangle", cgAngle
        return cgParticle



    def drawParticles(self, particleList, color):
        for point in particleList:
            weight = point.getWeight() * self.maxWeight
            b, g, r = color
            newColor = (b * weight, g * weight, r * weight)
            self.olinMap.drawPose(point, color = newColor)

    def weightList(self):
        """Builds and returns a list of the weights of the current valid particle list."""
        return [pose.getWeight() for pose in self.validPosList]



if __name__ == '__main__':
    # test = monteCarloLoc()
    # test.initializeParticles(1000)
    # # print "total len ", len(test.validPosList)
    # test.drawParticles(test.validPosList, (255,100,0))
    # cv2.waitKey(0)
    #
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
    #     cv2.imshow("Particles", test.currentMap)
    #     cv2.waitKey( 0)
    #
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    test = monteCarloLoc()
    test.initializeParticles(1)
    part = test.validPosList[0]
    part.setLoc(15.0,50.0,112)
    test.drawParticles((0,0,255))

    for i in range(50):
        test.particleMove((-1.0,-1.0,0.0))
        print test.validPosList[0]
        test.drawParticles((0,0,255))
        if len(test.validPosList) == 0:
            break
        cv2.waitKey(0)

    cv2.destroyAllWindows()


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
