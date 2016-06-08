""" ========================================================================
 * QRrecognizer.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 *  The ORBrecognizer object computes the similarity of objects using the ORB
 *  feature detector.
 *
========================================================================="""


import math
import random
import time
import os
import cv2
import numpy as np
import partitionAlg
import FeatureType
import FoxQueue
import OutputLogger
from operator import itemgetter
import turtleQR
import zbar


def qrScan(self, image):
        self.qrScanner.parse_config('enable')
        bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("bwimg", bwImg)
        #cv2.waitKey(0)
        pil_im = Image.fromarray(bwImg)
        pic2 = pil_im.convert("L")
        wid, hgt = pic2.size
        #print("wid, hgt,", wid, hgt)
        raw = pic2.tobytes()

        img = zbar.Image(wid, hgt, 'Y800', raw)
        result = self.qrScanner.scan(img)
        #print "RESULT", result
        if result == 0:
            #print "Scan failed"
            pass
        else:
            #print ("img is ", img)
            for symbol in img:
                #print "symbol did indeed get assigned"
                pass
            del(img)
            codeData = symbol.data.decode(u'utf-8')
            #print "Data found:", codeData
            list = string.split(codeData)
            #print(list)
            nodeNum = list[0]
            nodeCoord = list[1] + ' ' + list[2]
            nodeName = ''
            for i in range(3, len(list)):
                nodeName = nodeName + ' ' + list[i]
            
            #nodeNum, nodeCoord, nodeName = string.split(codeData)
            with self.lock:
                self.qrInfo = (int(nodeNum), nodeCoord, nodeName)
