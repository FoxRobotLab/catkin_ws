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
        """some conversion which should only be necessary when dealing with kinect images,
        but runs fine when given webcam images."""
        self.qrScanner.parse_config('enable')
        bwImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(bwImg)
        pic2 = pil_im.convert("L")
        width, height = pic2.size
        raw = pic2.tobytes()

        img = zbar.Image(width, height, 'Y800', raw)
        result = self.qrScanner.scan(img)
        if result == 0:
            #print "Scan failed"
            return None
        else:
            #just setting symbol to the last thing in img, which is where the data is stored because zbar is weird
            for symbol in img:
                pass
            del(img) #img not needed anymore, as data is in symbol
            codeData = symbol.data.decode(u'utf-8')
            #print "Data found:", codeData

            """this is specific to the format used for the QR data:
            nodeNumber coordinates name
            for example
            0 (22.2, 4.0) Home lab
            where coordinates has one space, in between the x and y coords, and the name may
            contain spaces. So first make sure it at least has all these things and the first
            thing is a number - so it's OK if it sees a QR code that isn't ours or sees something
            it thinks is a QR code but really isn't. Then format the output appropriately."""
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
