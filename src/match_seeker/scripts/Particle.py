"""=================================
File: Particle.py
Author: Susan Fox
Date: July, 2017

This defines a Particle class separate from the Monte Carlo Localization class, mostly so we could use it separately
for things like the map.
"""

import math
import numpy as np

class Particle():

    def __init__(self, mapObj, initPose = (0, 0, 0), mode = 'spec'):
        """Input is a global map object, an optional initial pose, and
        a mode. In "spec" mode the initial pose is used, and if the mode
        is 'random' then a new random pose is generated."""
        self.mapObj = mapObj
        # random perturbation constants: std dev for movement and turning as a percentage of movement
        # so if movement in x direction is 500 cm, then the std dev is 100
        # self.sigma_fwd_pct = 0.2
        # self.sigma_theta_pct = 0.05

        (self.x, self.y, self.heading) = initPose
        self.weight = 0.0
        if mode == 'random':
            self.setToRandom()


    def setToRandom(self):
        (xBound, yBound) = self.mapObj.getMapSize()
        while True:
            self.x = np.random.uniform(0, xBound)
            self.y = np.random.uniform(0, yBound)
            self.heading = np.random.uniform(0, 360)
            if self.isValid():
                return


    def scatter(self, x, y):
        """generates a new x, y in a uniform distribution around a given point and a random heading between 0-360"""
        range = 5.0
        while True:
            self.x = np.random.uniform(x-range, x+range)
            self.y = np.random.uniform(y-range, y+range)
            self.heading = np.random.uniform(0, 360)
            if self.isValid():
                return


    def makePerturbedCopy(self, scatter = False):
        """Makes a new copy that is nearby, but randomly perturbed using
        Gaussian distributions. Note: If enough attempts are made to generate
        a nearby copy and they fail, then a random particle is created."""

        if scatter == True:
            mult = 5
        else:
            mult = 1

        for i in range(10):
            newAngle = np.random.normal(self.heading, 3.0 * mult) % 360
            newX = np.random.normal(self.x, 0.25 * mult)
            newY = np.random.normal(self.y, 0.25 * mult)

            posParticle = Particle(self.mapObj, (newX, newY, newAngle))

            if posParticle.isValid():
                return posParticle
        else:
            return Particle(self.mapObj, mode = 'random')


    def isValid(self):
        """checks if the particle is in a valid location, using the map object's
        isAllowedLocation method."""

        return self.mapObj.isAllowedLocation(self.getLoc())


    def moveParticle(self, moveX, moveY, moveAngle):
        """Move the particle given a robot-centered x, y, and heading."""
        radHead = math.radians(self.heading)
        gx = moveX * math.cos(radHead) - moveY * math.sin(radHead)
        gy = moveX * math.sin(radHead) + moveY * math.cos(radHead)
        self.x += gx
        self.y += gy
        self.heading += moveAngle
        self.heading = self.heading % 360


    def calculateWeight(self, mclData):
        """Calculate the weight for this particle given the current input data."""
        poseTuple = (self.x, self.y, self.heading)
        minDist = self.mapObj.straightDist3d(mclData['odomPose'], poseTuple)
        minScore = mclData['odomScore']
        matchPoses = mclData['matchPoses']
        matchScores = mclData['matchScores']
        for m in range(len(matchPoses)):
            matchDist = self.mapObj.straightDist3d(matchPoses[m], poseTuple)
            if minDist > matchDist:
                minDist = matchDist
                minScore = matchScores[m]
        assert minDist >= 0
        self.weight = (90 - minDist)  # * (minScore / 100))  # append the maximum weight for each particle


    def normWeight(self, sumWeight):
        self.weight = self.weight/sumWeight


    def setLoc(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading % 360



    def getLoc(self):
        return self.x, self.y, self.heading


    def setWeight(self, weight):
        assert weight>=0
        self. weight = weight


    def getWeight(self):
        return self.weight

    def getScaledLoc(self):
        return (self.x * self.weight,
                self.y * self.weight,
                self.heading * self.weight)

    def getScaledX(self):
        return self.x*self.weight

    def getScaledY(self):
        return self.y*self.weight

    def getScaledAngle(self):
        return self.heading*self.weight

    def __str__(self):
        formatStr = "Particle info: ({0: 4.2f}, {1:4.2f}, {2:4.2f}, {3:4.2f}))"
        return formatStr.format(self.x, self.y, self.heading, self.weight)

