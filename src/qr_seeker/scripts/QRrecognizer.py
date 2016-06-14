""" ========================================================================
 * QRrecognizer.py
 *
 *  Created on: June 2016
 *  Author: mulmer
 *
 *  The QRrecognizer object tries to find a QR code in a given image.
 *
========================================================================="""


import cv2
import zbar
from PIL import Image
import string

class QRrecognizer():
    """Holds data about ORB keypoints found in the input picture."""
    def __init__(self, bot):
        self.robot = bot
        self.fHeight, self.fWidth, self.fDepth = self.robot.getImage()[0].shape
        self.qrScanner = zbar.ImageScanner()


    def qrScan(self, image):
            self.qrScanner.parse_config('enable')
            bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(bwImg)
            pic2 = pil_im.convert("L")
            wid, hgt = pic2.size
            raw = pic2.tobytes()

            img = zbar.Image(wid, hgt, 'Y800', raw)
            result = self.qrScanner.scan(img)
            if result == 0:
                return None
            else:
                for symbol in img:
                    pass
                del(img)
                codeData = symbol.data.decode(u'utf-8')
                list = string.split(codeData)
                if len(list) < 4 or not list[0].isdigit():
                    print "I saw a bad QR code!"
                    return None
                nodeNum = list[0]
                nodeCoord = list[1] + ' ' + list[2]
                nodeName = ''
                for i in range(3, len(list)):
                    nodeName = nodeName + ' ' + list[i]

                return (int(nodeNum), nodeCoord, nodeName)
