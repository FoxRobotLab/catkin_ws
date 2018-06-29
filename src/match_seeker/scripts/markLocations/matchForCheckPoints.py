"Authors: Malini Sharma and Jane Pellegrini"

class combineFiles(object):

    def __init__(self, locFile, imgNumFile, outputFile):

        """This function initializes the necessary variables for the class. These variables include the path to text
        file with the image number/the time it was obtained, the path to the text file with the click number/time stamp
        /x coordinate/y coordinate/yaw, the list of lines from the location file, the list of lines from the image
        number file, the final list of lists that contain the image number/ x coordinate/y coordinate/yaw, and the path
        to the text file that the data will be written to."""

        self.locFile = locFile
        self.imgNumFile = imgNumFile

        self.locList = []
        self.imgNumList = []

        self.finalList = []

        self.outputFile = outputFile

    def setUpLocList(self):

        """This function sets up the location file so that it can be manipulated by the program. It opens up the
        location file, reads in the lines, and then closes the file. Then, this function splits each line into its
        individual elements, and replaces the time element with the time in seconds"""

        locs = open(self.locFile, "r")
        locsLines = locs.readlines()
        locs.close()

        for line in locsLines:
            element = line.split()
            time = element[1]
            newTime = self.convertTimeToSeconds(time)
            element[1] = newTime
            self.locList.append(element)

    def setUpImgList(self):

        """This function sets up the image number file so that it can be manipulated by the program. It opens up the
        image number file, reads in the lines, and then closes the file. Then, this function splits each line into its
        individual elements, and replaces the time element with the time in seconds"""

        imgs = open(self.imgNumFile, "r")
        imgsLines = imgs.readlines()
        imgs.close()

        for line in imgsLines:
            element = line.split()
            time = element[1]
            newTime = self.convertTimeToSeconds(time)
            element[1] = newTime
            self.imgNumList.append(element)

    def convertTimeToSeconds(self, timeString):
        """
        Takes in a string of time in the format HR:MN:S and returns the total number of seconds

        :param timeString:
        :return:
        """
        timeList= timeString.split(":")
        secondsFromHours = int(timeList[0]) * 60 * 60
        secondsFromMinutes = int(timeList[1]) * 60
        totalTime = secondsFromHours + secondsFromMinutes + int(timeList[2]);
        totalTime = int(totalTime)
        return(totalTime)

    def compareTimes(self):

        """This function actually compares the times between the two files. If two times are the same, the image
         number/x coordinate/y coordinate/yaw are put into a new list. This new list is then added to the final list
         of entries that need to be written to the output file."""

        for locLine in self.locList:
            for imgLine in self.imgNumList:
                intLocTime = int(locLine[1])
                intImgTime = int(imgLine[1])
                if intLocTime == intImgTime:
                    newList = []
                    newList.append(imgLine[0])
                    newList.append(locLine[2])
                    newList.append(locLine[3])
                    newList.append(locLine[4])
                    self.finalList.append(newList)

    def _writeData(self):

        """This function writes the data in the final list to the output text file."""

        fileOpen = False
        logFile = None
        try:
            logFile = open(self.outputFile, 'w')
            fileOpen = True
        except:
            print ("FAILED TO OPEN DATA FILE")

        for line in self.finalList:
            dataStr = str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + " " + str(line[3]) + "\n"
            if fileOpen:
                logFile.write(dataStr)
        logFile.close()

    def go(self):

        """"This function just runs all the component functions so that the class can execute."""

        self.setUpLocList()
        self.setUpImgList()
        self.compareTimes()
        self._writeData()


if __name__ == '__main__':

    combiner = combineFiles(locFile="/home/macalester/Data-Jun28Thu-11_41_31.txt",
                        imgNumFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/testTurtlebotVidFrames/testTurtlebotVidFrames.txt",
                        outputFile="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/combiningTextFiles.txt")

    combiner.go()
