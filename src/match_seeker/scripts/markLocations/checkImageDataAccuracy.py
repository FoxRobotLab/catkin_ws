# Summer 2019. Cycle through image data to verify accuracy of headings, x, y, and cell tags. Writes new file tagged with date/time that copies over approved/changed data.
# author: Avik Bosshardt, Angel Sylvester, Maddie AlQatami

import os
import cv2
from datetime import datetime

class ImageDataChecker(object):
    def __init__(self, dataSource, picNum=0):

        self.picNum = picNum
        self.imgFileList = os.listdir(dataSource)
        self.imgFileList.sort()
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
        with open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morecells.txt",'r') as cellfile:
            lines = cellfile.readlines()
            for line in lines:
                line = line.strip('\n')
                array = line.split(" ")
                self.image_ch_dict[int(array[0])] = (array[1],array[2])

        with open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morelocs.txt",'r') as cellfile:
            lines = cellfile.readlines()
            for line in lines:
                line = line.strip('\n')
                array = line.split(" ")
                self.image_xyh_dict[int(array[0])] = (array[1],array[2],array[3])


    def slideshow(self):
        newMoreLocs = open(
            "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morelocs-" + datetime.now().strftime(
                "%m-%d-%Y %H-%M-%S") + ".txt", 'w')
        newMoreCells = open(
            "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morecells-" + datetime.now().strftime(
                "%m-%d-%Y %H-%M-%S") + ".txt", 'w')
        for file in self.imgFileList:
            self.picNum = int(self._extractNum(file))
            print(self.picNum)
            img = cv2.imread(self.dataSource+file)
            print(self.image_xyh_dict[self.picNum])
            print(self.image_ch_dict[self.picNum])
            cv2.imshow("Image",img)
            cv2.waitKey(50)
            prompt = raw_input("'k' for keep, 'f' for fix 'q' for quit: ")

            if prompt == 'k':
                newMoreLocs.writelines(str(self.picNum) + ' ' + self.image_xyh_dict[self.picNum][0] + " " +
                                       self.image_xyh_dict[self.picNum][1] + " " + self.image_xyh_dict[self.picNum][2] + '\n')
                newMoreCells.writelines(
                    str(self.picNum) + ' ' + self.image_ch_dict[self.picNum][0] + " " +
                    self.image_ch_dict[self.picNum][1] + '\n')

            elif prompt == 'f':
                prompt = raw_input("Heading? y/n: ")
                if prompt == "y":
                    heading = raw_input("Enter new heading: ")
                else:
                    heading = self.image_ch_dict[self.picNum][1]
                prompt = raw_input("Other errors? y/n: ")
                if prompt == "y":
                    xyc = raw_input("Enter x y cell: ").split(" ")
                else:
                    xyc = [self.image_xyh_dict[self.picNum][0], self.image_xyh_dict[self.picNum][1], self.image_ch_dict[self.picNum][0]]

                newMoreLocs.writelines(str(self.picNum) + ' ' + xyc[0] + " " + xyc[1] + " " + heading + '\n')
                newMoreCells.writelines(
                    str(self.picNum) + ' ' + xyc[2] + " " +
                    heading + '\n')
            elif prompt == 'q':
                break
            else:
                continue
        cv2.destroyAllWindows()
        newMoreLocs.close()
        newMoreCells.close()



    def checkforDuplicates(self):
        numList = []
        print("CELL DICT")
        for i in range(len(self.image_ch_dict.keys())-1):
            num1 = self.image_ch_dict.keys()[i]
            numList.append(int(num1))
        numList.sort()

        for i in range(len(numList)-1):
            num1 = numList[i]
            num2 = numList[i+1]


            if int(num2) - int(num1) != 1:
                print("problem at", num1)
        print("XYH DICT")
        numList = []
        for i in range(len(self.image_xyh_dict.keys()) - 1):
            num1 = self.image_xyh_dict.keys()[i]
            numList.append(int(num1))
        numList.sort()

        for i in range(len(numList) - 1):
            num1 = numList[i]
            num2 = numList[i + 1]

            if int(num2) - int(num1) != 1:
                print("problem at", num1)


    def _extractNum(self, fileString):
        """finds sequence of digits"""

        numStr = ""
        foundDigits = False
        for num in fileString:
            if num in '0123456789':
                foundDigits = True
                numStr += num
            elif foundDigits == True:
                break
        if numStr != "":
            return numStr
        else:
            self.picNum += 1
            return self.picNum


IMAGE_PATH = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/moreframes/"

starting_image_number = raw_input("Enter a starting image number:\n")

test = ImageDataChecker(dataSource=IMAGE_PATH,picNum=starting_image_number)
#test.checkforDuplicates()
test.slideshow()

