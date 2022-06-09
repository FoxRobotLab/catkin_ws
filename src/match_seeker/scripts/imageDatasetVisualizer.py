
import cv2
from OlinWorldMap import WorldMap
import ImageDataset
from OutputLogger import OutputLogger
from DataPaths import basePath, imageDirectory, locData

class DatasetVisualizer(object):


    def __init__(self):
        """"""
        self.worldMap = WorldMap()
        self.logger = OutputLogger(True, True)
        self.fakeGUI = FakeGUI()
        self.dataset = ImageDataset.ImageDataset(self.logger, self.fakeGUI)
        self.dataset.setupData(basePath + imageDirectory, basePath + locData, "frames", "jpg")


    def countWithinRange(self, xPos, yPos, radiusInMeters):
        """Counts all images in the dataset whose position is within the circle centered on (xPos, yPos)
        that is radiusInMeters in radius"""
        closePics = self.dataset.getNearPos((xPos, yPos), radiusInMeters)
        return len(closePics)


    def colorMapDensity(self):
        for x in range(5, 35, 2):
            for y in range(5, 65, 2):
                numPics = self.countWithinRange(x, y, 2.0)
                mapColor = self.mapCountToColor(numPics)
                self.worldMap.drawPose((x, y, 0), 4, mapColor)
                self.worldMap.displayMap()

    def mapCountToColor(self, picCount):
        """Map the picCount to a color, with red has highest and blue as smallest"""
        if picCount >= 100:
            return (0, 0, 255)  # red
        elif picCount >= 50:
            return (0, 255, 255)  # yellow
        elif picCount >= 25:
            return (0, 255, 0)  # green
        elif picCount >= 10:
            return (255, 0, 0)  # blue
        elif picCount > 0:
            return (0, 0, 0)  # black
        else:
            return (255, 255, 255)



class FakeGUI(object):
    def __init__(self):
        pass


    def updateRadius(self, x):
        pass

    def updateMessageText(self, X):
        pass



if __name__ == '__main__':
    dv = DatasetVisualizer()
    minVal = 1000
    maxVal = -1
    histDict = {0: 0, 100: 0, 200: 0, 300: 0, 400: 0, 500: 0}
    for x in range(5, 35, 2):
        for y in range(5, 65, 2):
            numPics = dv.countWithinRange(x, y, 2.0)
            if numPics >= 100:
                print "LOTS OF PICS", x, y
            print (x, y), numPics
            if numPics < minVal:
                minVal = numPics
            if numPics > maxVal:
                maxVal = numPics
            bin = (numPics // 100) * 100
            if bin > 500:
                bin = 500
            histDict[bin] += 1
    print 'MIN:', minVal, "   MAX:", maxVal
    for i in [0, 100, 200, 300, 400, 500]:
        print(i, histDict[i])

    dv.colorMapDensity()
    cv2.waitKey(0)


