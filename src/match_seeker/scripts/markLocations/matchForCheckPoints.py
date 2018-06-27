import os
import cv2
import src.match_seeker.scripts.markLocations.readMap as readMap
import numpy as np
import time
import math

class Matcher(object):

    def __init__(self, walkerTextFile, autoTextFile, outputTextFile):
        self.walkerTextFile = walkerTextFile
        self.autoTextFile = autoTextFile
        self.outputTextFile = outputTextFile


    def findMatches(self):
        """
        Writes a text file with the position given by the user written next to the index of the list in the auto text
        file.

        :return:
        """
        walker, walkerTimes = self.prepareWalker()
        auto, autoTimes = self.prepareAuto()

        output = open(self.outputTextFile, "w")
        for line in walker:
            elem = line.split()
            autoIndex = self.compareWithAllAutoTimes(elem[1], autoTimes)
            output.write(str(autoIndex) + " " + str(elem[2] + " " + elem[3] + " " + elem[4] + "\n"))
        output.close()




    def prepareWalker(self):
        """
        Reads the text file of all of the frames saved by the person walking, reads them. Returns two lists, the first
        containing all of the times, the second containing all of the data.

        :return:
        """
        timeList = []
        walker = open(self.walkerTextFile, "r")
        walkerLines = walker.readlines()
        walker.close()
        for line in walkerLines:
            elems= line.split()
            timeList.append(self.convertTimeToSeconds(elems[1]))
        return timeList, walkerLines

    def prepareAuto(self):
        """
        Reads the text file of all of frames saved by the computer, reads them. Returns two lists, the first containing
        all of the times, the second containing all of the data.

        :return:
        """
        timeList = []
        auto = open(self.autoTextFile, "r")
        autoLines = auto.readlines()
        auto.close()
        for line in autoLines:
            elems = line.split()
            timeList.append(self.convertTimeToSeconds(elems[1]))
        return timeList, autoLines



    def compareWithAllAutoTimes(self, walkerTime, autoTimes):
        """
        Finds the index in the auto list which is the most similar to the given walkerTime

        :param walkerTime:
        :param autoTimes:
        :return:
        """
        minDif = np.inf
        difIndex =0
        for i in range(len(autoTimes)):
            autoTime = autoTimes[i]
            diff = abs(autoTime - walkerTime)
            if diff < minDif:
                minDif = diff
                difIndex =i
        return difIndex






    def convertTimeToSeconds(self, timeString):
        """
        Takes in a string of time in the format HR:MN:S and returns the total number of seconds

        :param timeString:
        :return:
        """
        timeList= timeString.split(":")
        secondsFromHours = int(timeList[0]) * 60 * 60
        secondsFromMinutes = int(timeList[1]) * 60
        totalTime = secondsFromHours + secondsFromMinutes + int(timeList[2])
        return(totalTime)







if __name__ == "__main__":
    matchWalkerAndImages = Matcher(walkerTextFile= "",
                                   autoTextFile = "",
                                   outputTextFile = "")

