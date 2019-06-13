# Summer 2019. Cycle through image data to verify accuracy of headings, x, y, and cell tags. Writes new file tagged with date/time that copies over approved/changed data.
# author: Avik Bosshardt, Angel Sylvester, Maddie AlQatami

#fun fact: frames 23836-23842, 24168-24187 are missing from ...frames/moreframes

import os
import cv2
from datetime import datetime

class ImageDataChecker(object):
    def __init__(self, dataSource, picNum=0):

        self.picNum = int(picNum)
        fileList = os.listdir(dataSource)
        self.imgFileList = [f for f in fileList if f.endswith(".jpg")]
        print("Number of images:", len(self.imgFileList))
        self.imgFileList.sort(key=lambda s: int(self._extractNum(s)))
        self.setupImgFileList()
        self.dataSource = dataSource

        self.image_ch_dict = {}
        self.image_xyh_dict = {}
        self.readFiles()

    def setupImgFileList(self):
        for i in range(len(self.imgFileList)):
            if int(self._extractNum(self.imgFileList[i])) == int(self.picNum):
                self.imgFileList = self.imgFileList[i:]
                break

    def readFiles(self):
        with open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/frameHeadingFile_153_158.txt",'r') as cellfile:
            lines = cellfile.readlines()
            for line in lines:
                if line == 'NEW SET HERE\n':
                    break
                line = line.strip('\n')
                array = line.split(" ")
                self.image_ch_dict[int(array[0])] = (array[1],array[4])
                print line

        with open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/frame-xyh",'r') as cellfile:
            print 'hello'
            lines = cellfile.readlines()
            for line in lines:
                if line == 'NEW SET HERE\n':
                    break
                line = line.strip('\n')
                array = line.split(" ")
                self.image_xyh_dict[int(array[0])] = (array[1],array[2],array[3])
                print line


    def slideshow(self):
        newMoreLocs = open(
            "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/morelocs-" + datetime.now().strftime(
                "%m-%d-%Y %H-%M-%S") + ".txt", 'w')
        newMoreCells = open(
            "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/morecells-" + datetime.now().strftime(
                "%m-%d-%Y %H-%M-%S") + ".txt", 'w')
        for file in self.imgFileList:
            if file.endswith(".jpg"):
                self.picNum = int(self._extractNum(file))
                print self.picNum
                img = cv2.imread(self.dataSource+file)
                print "(x, y) =", [float(x) for x in self.image_xyh_dict[self.picNum]]
                print "theta =", [int(float(x)) for x in self.image_ch_dict[self.picNum]]
                head = int(float(self.image_ch_dict[self.picNum][1]))
                imgCopy = img.copy()
                cv2.putText(imgCopy, str(head), (280, 420), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,0))
                cv2.imshow("Image",imgCopy)
                x = cv2.waitKey(0)
                userChoice = chr(x & 0xFF)
                # userChoice = raw_input("k for keep, f for fix, q for quit: ")
                fixOptions = ['u', 'i', 'o', 'j', 'l', 'm', ',', '.']

                if userChoice == ' ':
                    headStr = self.image_xyh_dict[self.picNum][2]
                    heading = str(int(float(headStr)))
                    newMoreLocs.writelines(str(self.picNum) + ' ' + self.image_xyh_dict[self.picNum][0] + " " +
                                           self.image_xyh_dict[self.picNum][1] + " " + heading + '\n')
                    newMoreCells.writelines(
                        str(self.picNum) + ' ' + self.image_ch_dict[self.picNum][0] + " " +
                        heading + '\n')
                #elif userChoice == 'f':

                elif fixOptions.__contains__(userChoice):
                    # prompt = raw_input("Heading? y/n: ")
                    # if prompt == "y":
                    #x = cv2.waitKey(0)
                    #headKey = chr(x & 0xFF)   # raw_input("Enter new heading: ")
                    heading = self._makeHeading(userChoice)
                    print "New heading =", heading
                    # else:
                    #     heading = self.image_ch_dict[self.picNum][1]
                    # prompt = raw_input("Other errors? y/n: ")
                    # if prompt == "y":
                    #     xyc = raw_input("Enter x y cell: ").split(" ")
                    # else:
                    xyc = [self.image_xyh_dict[self.picNum][0], self.image_xyh_dict[self.picNum][1], self.image_ch_dict[self.picNum][0]]

                    newMoreLocs.writelines(str(self.picNum) + ' ' + xyc[0] + " " + xyc[1] + " " + heading + '\n')
                    newMoreCells.writelines(
                        str(self.picNum) + ' ' + xyc[2] + " " +
                        heading + '\n')
                    imgCopy = img.copy()
                    cv2.putText(imgCopy, heading, (280, 420), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0))
                    cv2.imshow("Image", imgCopy)
                    cv2.waitKey(50)

                elif userChoice == 'q':
                    break
                else:
                    print("========You typed -->" + userChoice + "<--- and I don't know what to do with that")
        cv2.destroyAllWindows()
        newMoreLocs.close()
        newMoreCells.close()

    def _makeHeading(self, headStr):
        """Given a string from uio, jkl, m,., converts to the relevant direction."""
        if headStr == 'u':
            return '45'
        elif headStr == 'i':
            return '0'
        elif headStr == 'o':
            return '315'
        elif headStr == 'j':
            return '90'
        elif headStr == 'l':
            return '270'
        elif headStr == 'm':
            return '135'
        elif headStr == ',':
            return '180'
        elif headStr == '.':
            return '225'



    def checkforDuplicates(self):
        numList = []
        print("CELL DICT")
        for i in range(len(self.image_ch_dict.keys())):
            num1 = self.image_ch_dict.keys()[i]
            numList.append(int(num1))
        numList.sort()

        for i in range(len(numList)-1):
            num1 = numList[i]
            num2 = numList[i+1]


            if int(num2) - int(num1) != 1:
                print num1
                print num2
                print("problem at", num1)
        print("XYH DICT")
        numList = []
        for i in range(len(self.image_xyh_dict.keys())):
            num1 = self.image_xyh_dict.keys()[i]
            numList.append(int(num1))
        numList.sort()

        for i in range(len(numList) - 1):
            num1 = numList[i]
            num2 = numList[i + 1]

            if int(num2) - int(num1) != 1:
                print num1
                print num2
                print("problem at", num1)


    def _extractNum(self, fileString):
        """finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for ch in fileString:
            if ch in '0123456789':
                foundDigits = True
                numStr += ch
            elif foundDigits == True:
                break
        if numStr != "":
            return numStr
        # else:
        #     self.picNum += 1
        #     return self.picNum


IMAGE_PATH = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/06-11-2019-13-56-55newframes/"

starting_image_number = raw_input("Enter a starting image number:\n")

test = ImageDataChecker(dataSource=IMAGE_PATH,picNum=starting_image_number)
#test.checkforDuplicates()
test.slideshow()

