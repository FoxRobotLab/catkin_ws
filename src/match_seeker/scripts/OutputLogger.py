""" ========================================================================
OutputLogger.py

Created: March 24, 2016
Author: Susan Fox

An object that write to screen or log file, or both.
======================================================================== """


import time
import os
from DataPaths import basePath


class OutputLogger:
    """The output logger is told when created whether to write to screen,
    file, or both (changable with methods, too). Then when the program
    "logs" something, it does the right thing."""

    def __init__(self, toFile = True, toConsole = True):
        """If logging to a file, sets up the file with the right name based\
        on the current date and time."""

        self.toFile = toFile
        self.toConsole = toConsole
        self.directory = basePath + 'res/logs/'
        self.fileOpen = False
        self.logOpen = True

        if self.toFile:
            try:
                os.makedirs(self.directory)
            except:
                pass
            self.logName = time.strftime("%b%d%a-%H%M%S.txt")
            try:
                self.logFile = open(self.directory + self.logName, 'w')
                self.fileOpen = True
            except e:
                print "LOGGER FAILED TO OPEN LOG FILE"



    def log(self, line):
        """Writes the given line to file or console"""
        if self.logOpen:
            if self.toFile and self.fileOpen:
                self.logFile.write(line + '\n')
                self.logFile.flush()
            if self.toConsole:
                print line


    def close(self):
        """When called, closes the log file if one is open."""
        if self.fileOpen:
            self.logFile.close()
            self.fileOpen = False
        self.logOpen = False

    def reset(self):
        """Reset the log to open the old log and add to it?"""
        self.fileOpen = False
        self.logOpen = True

        if self.toFile:
            try:
                self.logFile = open(self.directory + self.logName, 'a')
                self.fileOpen = True
            except e:
                print "LOGGER FAILED TO OPEN LOG FILE"


if __name__ == '__main__':
    logger = OutputLogger(False, False)
    logger.log("First line")
    logger.log("Second line")
    for i in range(50):
        s = str(i) + str(i*5)
        logger.log(s)
    logger.close()
    logger.log("Shouldn't see this")
    logger.reset()
    logger.log("Third line")
