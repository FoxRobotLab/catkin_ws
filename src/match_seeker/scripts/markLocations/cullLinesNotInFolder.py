import os
import cv2
import numpy as np

"""
Author: Jane Pellegrini

This is designed for addressing the problem of having less frames than lines in your text file. This will comb through
both files and write a new text file that only has lines for frames that are in the folder
"""

class lineCuller(object):

    def __init__(self, imageFolderPath, textFilePath, outputTextFile):
        self.imageFolderPath= imageFolderPath
        self.textFilePath = textFilePath
        self.outputTextFile = outputTextFile

    def go(self):
        frameNames = os.listdir(self.imageFolderPath)
        #if filename[-3:] in {"jpg", "png"}:
        frameNums= []
        for frame in frameNames:
            if frame[-3:] in {"jpg", "png"}:
                frameNum = self._extractNum(frame)
                frameNums.append(frameNum)
        textFileRead= open(self.textFilePath, "r")
        txtFileLines = textFileRead.readlines()
        textFileRead.close()
        textFileNums = []
        for line in txtFileLines:
            elems = line.split()
            num = elems[0]
            textFileNums.append(int(num))

        newTextFile = open(self.outputTextFile, "w")
        for textNum in textFileNums:
            found = False
            for frameNum in frameNums:
                if frameNum == textNum:
                    found =True
            if found ==True:
                self.writeLine(textNum, txtFileLines,newTextFile)

    def writeLine(self, num, fileLines, newTextFile):
        for line in fileLines:
            elems = line.split()
            if num == elems[0]:
                newTextFile.write(line)









    def _extractNum(self, fileString):
        """finds sequence of digits"""
        numStr = ""
        foundDigits = False
        for c in fileString:
            if c in '0123456789':
                foundDigits = True
                numStr += c
            elif foundDigits == True:
                break
        if numStr != "":
            return int(numStr)


if __name__ == "__main__":
    cullerOfLines = lineCuller(imageFolderPath="",
                               textFilePath="",
                               outputTextFile="")
