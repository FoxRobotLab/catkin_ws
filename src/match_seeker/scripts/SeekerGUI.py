
import Tkinter as tk
import threading

class SeekerGUI():

    def __init__(self):
        """Sets up all the GUI elements"""
        self.mainWin = tk.Tk()
        self.mainWin.title("Match Seeker")

        self.mode = tk.StringVar()
        self.mode.set("Navigating")
        self.navType = tk.StringVar()
        self.navType.set("Images")

        self.odomList = []
        self.lastKnownList = []
        self.MCLList = []
        self.bestPicList = []
        self.secondPicList = []
        self.thirdPicList = []
        self.setupLocLists()

        self.messageText = "Booting up..."
        # self.messageText.set()

        self.turnState = tk.StringVar()
        self.turnState.set("turn status")
        self.turnInfo = []
        self.setupTurnList()
        self.tDist = tk.StringVar()
        self.tDist.set(0.0)

        self.cNode = tk.StringVar()
        self.cNode.set(1)
        self.radius = tk.StringVar()
        self.radius.set(15)
        self.matchStatusInfo = tk.StringVar()
        self.matchStatusInfo.set("I have no idea where I am")


        self.setUpHeadFrame()
        self.setUpLocFrame()
        self.setUpMessages()
        self.setUpturnInfo()
        self.setUpImgMatch()


    def setupLocLists(self):
        """each list has x, y, heading, and confidence"""
        for list in [self.odomList, self.lastKnownList, self.MCLList, self.bestPicList, self.secondPicList, self.thirdPicList]:
            for i in range(4):
                var = tk.StringVar()
                var.set(0.0)
                list.append(var)

    def setupTurnList(self):
        """a turn statement, currHead, targetHeading, angleToTurn, tDist"""
        for i in range(3):
            var = tk.StringVar()
            var.set(0.0)
            self.turnInfo.append(var)

    def setUpHeadFrame(self):
        # Heading frame
        headFrame = tk.Frame(self.mainWin, bg="pale green", bd=2, relief=tk.GROOVE)
        headFrame.grid(row = 0, column = 0, columnspan=4)
        modeLab = tk.Label(headFrame, bg="pale green", text="Mode:", font="DroidSans 24", width=8, height = 1)
        modeLab.grid(row = 0, column = 0)
        self.currentMode = tk.Label(headFrame, bg="spring green", textvariable=self.mode, font="DroidSans 24 bold", width=17, height =1)
        self.currentMode.grid(row = 0, column = 1)
        navLab = tk.Label(headFrame, bg="pale green", text="Nav by:", font="DroidSans 24", width=8, height =1)
        navLab.grid(row = 0, column = 2)
        self.currentNav = tk.Label(headFrame, textvariable=self.navType, bg="spring green", font="DroidSans 24 bold", width=17, height =1)
        self.currentNav.grid(row = 0, column = 3)


    def setUpLocFrame(self):
        #Locations frame
        locationsFrame = tk.Frame(self.mainWin, bg="light blue", bd=2, relief=tk.GROOVE)
        locationsFrame.grid(row = 1, column = 0)

        #row heading
        odomLabel = tk.Label(locationsFrame, bg="cornflower blue", text="Odometry Location:", font="DroidSans 16", width=20)
        odomLabel.grid(row = 1, column = 0)
        lastKnownLabel = tk.Label(locationsFrame, bg="cornflower blue", text="Last Known Loc:", font="DroidSans 16", width=20)
        lastKnownLabel.grid(row = 2, column = 0)
        MCLLabel = tk.Label(locationsFrame, bg="cornflower blue", text="MCL Center:", font="DroidSans 16", width=20)
        MCLLabel.grid(row = 3, column = 0)
        ClosestPic = tk.Label(locationsFrame, bg="cornflower blue", text="Best Match:", font="DroidSans 16", width=20)
        ClosestPic.grid(row = 4, column = 0)
        secondPic = tk.Label(locationsFrame, bg="cornflower blue", text="Second Match:", font="DroidSans 16", width=20)
        secondPic.grid(row = 5, column = 0)
        thirdPic = tk.Label(locationsFrame, bg="cornflower blue", text="Third Match:", font="DroidSans 16", width=20)
        thirdPic.grid(row = 6, column = 0)

        #column heading
        xLabel = tk.Label(locationsFrame, bg="cornflower blue", text="x", font="DroidSans 16", width=5)
        xLabel.grid(row = 0, column = 1)
        yLabel = tk.Label(locationsFrame, bg="cornflower blue", text="y", font="DroidSans 16", width=5)
        yLabel.grid(row = 0, column = 2)
        hLabel = tk.Label(locationsFrame, bg="cornflower blue", text="h", font="DroidSans 16", width=5)
        hLabel.grid(row = 0, column = 3)
        confLabel = tk.Label(locationsFrame, bg="cornflower blue", text="conf", font="DroidSans 16", width=5)
        confLabel.grid(row = 0, column = 4)

        #value display, odom info

        odomX = tk.Label(locationsFrame, textvariable=self.odomList[0], bg="light blue", font="DroidSans 16", width=5)
        odomX.grid(row = 1, column = 1)
        odomY = tk.Label(locationsFrame, textvariable=self.odomList[1], bg="light blue", font="DroidSans 16", width=5)
        odomY.grid(row = 1, column = 2)
        odomH = tk.Label(locationsFrame, textvariable=self.odomList[2], bg="light blue", font="DroidSans 16", width=5)
        odomH.grid(row = 1, column = 3)
        odomConf = tk.Label(locationsFrame, textvariable=self.odomList[3], bg="light blue", font="DroidSans 16", width=5)
        odomConf.grid(row = 1, column = 4)

        #value display, last know loc info
        lastKnownX = tk.Label(locationsFrame, textvariable=self.lastKnownList[0], bg="light blue", font="DroidSans 16", width=5)
        lastKnownX.grid(row = 2, column = 1)
        lastKnownY = tk.Label(locationsFrame, textvariable=self.lastKnownList[1], bg="light blue", font="DroidSans 16", width=5)
        lastKnownY.grid(row = 2, column = 2)
        lastKnownH = tk.Label(locationsFrame, textvariable=self.lastKnownList[2], bg="light blue", font="DroidSans 16", width=5)
        lastKnownH.grid(row = 2, column = 3)
        lastKnownConf = tk.Label(locationsFrame, textvariable=self.lastKnownList[3], bg="light blue", font="DroidSans 16", width=5)
        lastKnownConf.grid(row = 2, column = 4)

        #value display, MCl Center of Mass
        MCLX = tk.Label(locationsFrame, textvariable=self.MCLList[0], bg="light blue", font="DroidSans 16", width=5)
        MCLX.grid(row = 3, column = 1)
        MCLY = tk.Label(locationsFrame, textvariable=self.MCLList[1], bg="light blue", font="DroidSans 16", width=5)
        MCLY.grid(row = 3, column = 2)
        MCLH = tk.Label(locationsFrame, textvariable=self.MCLList[2], bg="light blue", font="DroidSans 16", width=5)
        MCLH.grid(row = 3, column = 3)
        MCLConf = tk.Label(locationsFrame, textvariable=self.MCLList[3], bg="light blue", font="DroidSans 16", width=5)
        MCLConf.grid(row = 3, column = 4)

        #value display, best pic
        bestPicX = tk.Label(locationsFrame, textvariable=self.bestPicList[0], bg="light blue", font="DroidSans 16", width=5)
        bestPicX.grid(row = 4, column = 1)
        bestPicY = tk.Label(locationsFrame, textvariable=self.bestPicList[1], bg="light blue", font="DroidSans 16", width=5)
        bestPicY.grid(row = 4, column = 2)
        bestPicH = tk.Label(locationsFrame, textvariable=self.bestPicList[2], bg="light blue", font="DroidSans 16", width=5)
        bestPicH.grid(row = 4, column = 3)
        bestPicConf = tk.Label(locationsFrame, textvariable=self.bestPicList[3], bg="light blue", font="DroidSans 16", width=5)
        bestPicConf.grid(row = 4, column = 4)

        #value display, second closest pic
        secondPicX = tk.Label(locationsFrame, textvariable=self.secondPicList[0], bg="light blue", font="DroidSans 16", width=5)
        secondPicX.grid(row = 5, column = 1)
        secondPicY = tk.Label(locationsFrame, textvariable=self.secondPicList[1], bg="light blue", font="DroidSans 16", width=5)
        secondPicY.grid(row = 5, column = 2)
        secondPicH = tk.Label(locationsFrame, textvariable=self.secondPicList[2], bg="light blue", font="DroidSans 16", width=5)
        secondPicH.grid(row = 5, column = 3)
        secondPicConf = tk.Label(locationsFrame, textvariable=self.secondPicList[3], bg="light blue", font="DroidSans 16", width=5)
        secondPicConf.grid(row = 5, column = 4)

        #value display, third closest pic
        thirdPicX = tk.Label(locationsFrame, textvariable=self.thirdPicList[0], bg="light blue", font="DroidSans 16", width=5)
        thirdPicX.grid(row = 6, column = 1)
        thirdPicY = tk.Label(locationsFrame, textvariable=self.thirdPicList[1], bg="light blue", font="DroidSans 16", width=5)
        thirdPicY.grid(row = 6, column = 2)
        thirdPicH = tk.Label(locationsFrame, textvariable=self.thirdPicList[2], bg="light blue", font="DroidSans 16", width=5)
        thirdPicH.grid(row = 6, column = 3)
        thirdPicConf = tk.Label(locationsFrame, textvariable=self.thirdPicList[3], bg="light blue", font="DroidSans 16", width=5)
        thirdPicConf.grid(row = 6, column = 4)



    def setUpMessages(self):
        # Messages frame
        messageFrame = tk.Frame(self.mainWin, bg="light goldenrod", bd = 2, relief=tk.GROOVE, width = 480, height = 160)
        messageFrame.grid(row = 2, column = 0)
        # messageFrame.grid_propagate(0)

        messLabel = tk.Label(messageFrame, bg="orange", text = "Messages:", font="DroidSans 22 bold", width=31)
        messLabel.grid(row = 0, column = 0, columnspan = 2)
        self.messages = tk.Text(messageFrame, bg="light goldenrod",  wrap = tk.WORD, font="DroidSans 15",
                                   width = 43, height = 5) # width=15 height = 5
        self.messages.insert('end',self.messageText)
        self.messages.grid(row = 1, column = 0)

        scrollbar = tk.Scrollbar(messageFrame)
        self.messages.config(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=1, sticky = 'ns')
        scrollbar.config(command=self.messages.yview)


    def setUpturnInfo(self):
        # Coordinate check frame
        turnCheckFrame = tk.Frame(self.mainWin, bg="pink", bd=2, relief=tk.GROOVE)
        turnCheckFrame.grid(row = 1, column = 1)
        coordCheck= tk.Label(turnCheckFrame, bg="pale violet red", text="Turn Info", font="DroidSans 22 bold", width=25)
        coordCheck.grid(row = 0, column = 0, columnspan = 2)

        # display info, turning information
        turn = tk.Label(turnCheckFrame, textvariable = self.turnState, bg="pink", font = "DroidSans 15", width =30)
        turn.grid(row = 1, column = 0, columnspan = 2, pady = 15)

        #display info, currHead
        currHead1 = tk.Label(turnCheckFrame, text= "currHead =", bg = "pink", font = "DroidSans 15", width = 10)
        currHead1.grid(row = 2, column = 0)
        currHead2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[0], bg="pink", font = "DroidSans 15", width = 10)
        currHead2.grid(row = 2, column = 1)

        # display info, targetAngle
        targetAngle1 = tk.Label(turnCheckFrame, text="tAngle =", bg="pink", font="DroidSans 15", width=10)
        targetAngle1.grid(row = 3, column = 0)
        targetAngle2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[1], bg="pink", font="DroidSans 15", width=10)
        targetAngle2.grid(row = 3, column = 1)

        # display info, AngleToTurn
        angleToTurn1= tk.Label(turnCheckFrame, text="turnAngle =", bg="pink", font="DroidSans 15", width=10)
        angleToTurn1.grid(row = 4, column = 0)
        angleToTurn2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[2], bg="pink", font="DroidSans 15", width=10)
        angleToTurn2.grid(row =4, column = 1)

        # display info, tDist
        tDist1= tk.Label(turnCheckFrame, text="tDist =", bg="pink", font="DroidSans 15", width=10)
        tDist1.grid(row = 5, column = 0)
        tDist2 = tk.Label(turnCheckFrame, textvariable=self.tDist, bg="pink", font="DroidSans 15", width=10)
        tDist2.grid(row =5, column = 1)


    def setUpImgMatch(self):
        # Image matching frame
        imageMatchFrame = tk.Frame(self.mainWin, bg="plum1", bd=2, relief=tk.GROOVE)
        imageMatchFrame.grid(row = 2, column = 1)
        imageMatchInfo = tk.Label(imageMatchFrame, bg="plum3", text="Image Matching Info", font="DroidSans 22 bold", width=25)
        imageMatchInfo.grid(row = 0, column = 0, columnspan = 2)

        #display info, closest node
        closestNode1 = tk.Label(imageMatchFrame, text="Closest Node: ", bg="plum1", font="DroidSans 15", width=15)
        closestNode1.grid(row = 1, column = 0)
        closestNode2 = tk.Label(imageMatchFrame, textvariable=self.cNode, bg="plum1", font="DroidSans 15", width=10)
        closestNode2.grid(row = 1, column = 1)

        #display info, search radius
        searchRadius1 = tk.Label(imageMatchFrame, text="Search Radius: ", bg="plum1", font="DroidSans 15", width=15)
        searchRadius1.grid(row = 2, column = 0)
        searchRadius2 = tk.Label(imageMatchFrame, textvariable=self.radius, bg="plum1", font="DroidSans 15", width=10)
        searchRadius2.grid(row = 2, column = 1)


        #display info, match status
        matchStatus1 = tk.Label(imageMatchFrame, text="Match Status: ", bg="plum1", font="DroidSans 15", width=15)
        matchStatus1.grid(row = 3, column = 0, pady =2)
        matchStatus2 = tk.Label(imageMatchFrame, textvariable=self.matchStatusInfo, bg="plum1", font="DroidSans 15", width=25, justify="right")
        matchStatus2.grid(row = 4, column = 0, columnspan=2, pady = 3)


    def updateOdomList(self,loc):
        for i in range(len(self.odomList)):
            self.odomList[i].set('%.2f'%loc[i])

    def updateLastKnownList(self,loc):
        for i in range(len(self.lastKnownList)):
            self.lastKnownList[i].set('%.2f'%loc[i])

    def updateMCLList(self,loc):
        for i in range(len(self.MCLList)):
            self.MCLList[i].set('%.2f'%loc[i])

    def updatePicLocs(self,loc1, loc2, loc3):
        for i in range(len(loc1)):
            self.bestPicList[i].set('%.2f'%loc1[i])
            self.secondPicList[i].set('%.2f'%loc2[i])
            self.thirdPicList[i].set('%.2f'%loc3[i])

    def updatePicConf(self,scores):
        self.bestPicList[3].set('%.2f'%scores[0])
        self.secondPicList[3].set('%.2f'%scores[1])
        self.thirdPicList[3].set('%.2f'%scores[2])

    def updateMessageText(self,text):
        self.messages.insert('1.0',text+"\n")

    def updateTurnState(self,statement):
        self.turnState.set(statement)

    def updateTurnInfo(self,turnData):
        for i in range(len(turnData)):
            self.turnInfo[i].set('%.2f'%turnData[i])

    def updateTDist(self,dist):
        self.tDist.set('%.2f'%dist)

    def updateCNode(self,closestNode):
        self.cNode.set(closestNode)

    def updateRadius(self,radius):
        self.radius.set(radius)

    def updateMatchStatus(self,status):
        self.matchStatusInfo.set(status)

    def updateNavType(self,type):
        self.navType.set(type)

    def navigatingMode(self):
        self.mode.set("Navigating")

    def localizingMode(self):
        self.mode.set("Localizing")

    def stop(self):
        self.mainWin.destroy()

    def update(self):
        try:
            self.mainWin.update_idletasks()
            self.mainWin.update()
        except tk.TclError:
            pass

# if __name__ == '__main__':
#     gui = SeekerGUI()
#     gui.go()
