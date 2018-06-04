

class stripText(object):
    def __init__(self, origTextFileName, newTextFileName):
        self.origTextFileName = origTextFileName
        self.newTextFileName = newTextFileName
        self.toBeAdded = []
        self.beingAdded = []

    def readDeleteTextLines(self):
        file = open(self.origTextFileName, "r")
        while True:
            line = file.readline()
            line = line.rstrip()
            if (line.startswith("(")):
                self.toBeAdded.append(line)
            if not line: break


    def stripTheStrings(self):
        newFile = open(self.newTextFileName, "w")
        count = 0
        for elem in self.toBeAdded:
            count +=1
            newFile.write(str(count))
            lineStrings = elem.split(",")
            xCoord = lineStrings[3].replace("(", "")
            newFile.write(xCoord)
            yCoord = lineStrings[4]
            newFile.write(yCoord)
            yaw = lineStrings[5].replace(")", "")
            newFile.write(yaw + '\n')







if __name__ == "__main__":
    stripper = stripText(origTextFileName="/home/macalester/catkin_ws/src/match_seeker/res/locdata/badly_formatted_data.txt",
                        newTextFileName="/home/macalester/catkin_ws/src/match_seeker/res/locdata/Data-June01Fri-999999.txt" )
    stripper.readDeleteTextLines()
    stripper.stripTheStrings()

