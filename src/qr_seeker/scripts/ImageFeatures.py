""" ========================================================================
ImageFeatures.py

Created: May, 2016
Author: Susan

This program contains the features from images, including Hough Lines, and
color signatures (and watershed segmentation?).

NOTE:

* Hough lines similarity ranges between 0 and 200 for the test datasets, with
real similarity values below 100

* Color signature similarity values range between 0 and 2500, with real
similarity below 1000.0

======================================================================== """



import cv2

# import HoughLines
# import ImageSegmentation
# import ColorSignature
import OutputLogger
import ORBFeatures


class ImageFeatures:
    """Collects up different features of the image and can display the results, or compare to another for similarity."""

    def __init__(self, image, idNum, logger, orbFinder):
        """Initializes the ImageFeatures object with the image and output logger.
        Eventually will add in the actual features."""

        self.image = image.copy()
        self.height, self.width, self.depth = self.image.shape
        self.idNum = idNum
        self.logger = logger
        # self.houghLines = HoughLines.HoughLines(self.image, logger)
        # self.colorSignature = ColorSignature.ColorSignature(self.image, logger)
        self.ORBFeatures = ORBFeatures.ORBFeatures(self.image, logger, orbFinder)



    def copy(self):
        """Makes and returns a copy of itself.  JUST A STUB FOR NOW."""
        imFeat = ImageFeatures(self.image, self.idNum, self.logger)
        return imFeat


    def getIdNum(self):
        """Returns the id number associated with the image"""
        return self.idNum

    def getImage(self):
        """Returns the image associated with the features."""
        return self.image


    def displayFeaturePics(self, baseWindowName, startX, startY):
        """Displays a set of windows showing the relevant features, based on the
        base window name provided, and the starting x and y for the series of pictures."""
        #houghName = "Hough-Lines-" + baseWindowName
        vertOffset = self.height + 10
        #self.houghLines.displayFeaturePics(houghName, startX, startY + vertOffset)
        sigName = "ColorSig-" + baseWindowName
        #self.colorSignature.displayFeaturePics(sigName, startX, startY + 2 * vertOffset)
        orbName = "ORB-" + baseWindowName
        self.ORBFeatures.displayFeaturePics(sigName, startX, startY +  vertOffset)


    def evaluateSimilarity(self, featureObj):
        """Evaluate similarity based on features. Compares Hough Lines and Color Signatures."""
        #houghSim = self.houghLines.evaluateSimilarity(featureObj.houghLines)
        #self.logger.log("Hough Lines sim = " + str(houghSim))
        #colorSim = self.colorSignature.evaluateSimilarity(featureObj.colorSignature)
        orbSim = self.ORBFeatures.evaluateSimilarity(featureObj.ORBFeatures)
        #self.logger.log("ColorSig sim =" + str(colorSim))
        return orbSim #houghSim + colorSim + orbSim
