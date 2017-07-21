""" ========================================================================
drawOdom.py

This draws the odometry data on a map of olin rice.

======================================================================== """

#!/usr/bin/env python

from odometryListener import odometer
import rospy
import cv2
import numpy as np
from markLocations import readMap
from DataPaths import basePath



class drawOdom():

    def __init__(self):
        self.odom = odometer()
        self.odom.resetOdometer()
        self.mapFile = basePath + "res/map/olinNewMap.txt"
        self.startX = 22.2
        self.startY = 6.5
        self.startYaw = 0.0


    def run(self):
        self.getOlinMap()
        inx = 0
        while not rospy.is_shutdown():
            inx += 1
            x, y, yaw = self.odom.getOdomData()
            if inx == 50:
                inx = 0
                print "to get data"
                print x, y, yaw
            if not((x==None) or (y == None) or (yaw == None)):
                cv2.imshow("Map",self.drawLocOnMap(x,y,yaw))
                cv2.waitKey(20)
        cv2.destroyAllWindows()
        self.odom.exit()


    def getOlinMap(self):
        """Read in the Olin Map and return it. Note: this has hard-coded the orientation flip of the particular
        Olin map we have, which might not be great, but I don't feel like making it more general. Future improvement
        perhaps."""
        origMap = readMap.createMapImage(self.mapFile, 20)
        map2 = np.flipud(origMap)
        self.olinMap = np.rot90(map2)
        (self.mapHgt, self.mapWid, self.mapDep) = self.olinMap.shape


    def drawLocOnMap(self, offsetX, offsetY, offsetYaw):
        """Draws the current location on the map"""
        # positionTemplate = "({1:5.2f}, {2:5.2f}, {3:f})"
        # offsetTemplate = "(Offsets: {0:5.2f}, {1:5.2f}, {2:f})"
        nextMapImg = self.olinMap.copy()
        (pixX, pixY) = self.convertWorldToMap(self.startX + offsetX, self.startY + offsetY)
        self.drawPosition(nextMapImg, pixX, pixY, self.startYaw + offsetYaw, (255, 0, 0))
        # posInfo = positionTemplate.format(self.startX, self.startY, self.startYaw)
        # offsInfo = offsetTemplate.format(offsetX, offsetY, offsetYaw)
        # cv2.putText(nextMapImg, posInfo, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(nextMapImg, offsInfo, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        return nextMapImg


    def drawPosition(self, image, x, y, heading, color):
        """Draws the particle's position on the map."""
        cv2.circle(image, (x, y), 6, color, -1)
        newX = x
        newY = y
            #TODO: draw any heading
        cv2.line(image, (x, y), (newX, newY), color)


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



if __name__ == "__main__":
    rospy.init_node('Report Odometry')
    draw = drawOdom()
    draw.run()
    rospy.spin()
