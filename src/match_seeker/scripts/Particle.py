"""=================================
File: Particle.py
Author: Susan Fox
Date: July, 2017

This defines a Particle class separate from the Monte Carlo Localization class, mostly so we could use it separately
for things like the map.
"""

import math


class Particle():

    def __init__(self, x, y, heading):

        self.x = x
        self.y = y
        self.heading = heading

        self.weight = 0.0


    def moveParticle(self, moveX, moveY, moveAngle):

    #     self.x += moveX
    #     self.y += moveY
    #
    #     self.heading = self.heading + moveAngle
    #     if self.heading > 2 * np.pi:
    #         self.heading -= 2 * np.pi


        oldHead =  self.heading
        gx = moveX * math.cos(self.heading) + moveY * math.sin(self.heading)
        gy = moveX * math.sin(self.heading) + moveY * math.cos(self.heading)

        self.x += gx
        self.y += gy

        self.heading += moveAngle
        self.heading = self.heading % (2*np.pi)

    def normWeight(self, sumWeight):
        self.weight = self.weight/sumWeight


    def setLoc(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading


    def getLoc(self):
        return self.x, self.y, self.heading


    def setWeight(self, weight):
        self. weight = weight


    def getWeight(self):
        return self.weight


    def getScaledX(self):
        return self.x*self.weight

    def getScaledY(self):
        return self.y*self.weight

    def getScaledAngle(self):
        return self.heading*self.weight

