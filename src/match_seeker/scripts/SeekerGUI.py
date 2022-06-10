"""
SeekerGUI.py

This is a user interface that displays the information instead of the terminal.

All the methods start with setUp is the frames setup and all the methods
start with update can update the information on display.

"""
import Tkinter as tk
import threading

class SeekerGUI():

    def __init__(self):
        """Sets up all the GUI elements"""
        self.mainWin = tk.Tk()
        self.mainWin.title("Match Seeker")
        self.mainWin.geometry("+550+600")

        self.mode = tk.StringVar()
        self.mode.set("Navigating")
        self.navType = tk.StringVar()
        self.navType.set("Images")
        # self.mclColor = tk.StringVar()
        # self.mclColor.set("light blue")
        # self.imageColor = tk.StringVar()
        # self.imageColor.set("light blue")
        self.MCLConf = None
        self.bestPicConf = None

        self.userInputStartLoc = tk.StringVar()
        self.userInputStartYaw = tk.StringVar()
        self.confirmClickedStart = False
        self.userInputDest = tk.StringVar()
        self.confirmClickedDest = False

        self.userInputStartNode = tk.StringVar()
        self.userInputStartX = tk.StringVar()
        self.userInputStartY = tk.StringVar()




        self.odomList = []
        self.lastKnownList = []
        self.MCLList = []
        self.bestPicList = []
        self.secondPicList = []
        self.thirdPicList = []
        self.setupLocLists()

        self.messageText = "Booting up..."
        self.oddMessColor = True
        self.turnState = tk.StringVar()
        self.turnState.set("turn status")
        self.turnInfo = []
        self.setupTurnList()
        self.currHead2 = None
        self.targetAngle2 = None
        self.angleToTurn2 = None

        self.tDist = tk.StringVar()
        self.tDist.set(0.0)
        self.cNode = tk.StringVar()
        self.cNode.set(1)
        self.nextNode = tk.StringVar()
        self.nextNode.set(-1)
        self.radius = tk.StringVar()
        self.radius.set(15)
        self.matchStatusInfo = tk.StringVar()
        self.matchStatusInfo.set("I have no idea where I am")


        self.setUpHeadFrame()
        self.setUpLocFrame()
        self.setUpMessages()
        self.setUpturnInfo()
        self.setUpImgMatch()

    def askWhich(self):
        """
        The pop up window that asks whether the user will enter the node or the coordinates of the starting
        position
        """
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input start node/location (space in between x and y)")
        popupWin.geometry("+550+400")
        self.userInputStartLoc.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popupWin, bg="gray22", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        nodeButton = tk.Button(popupFrame, bg="indian red", fg="snow", text="Node", command= lambda: [popupWin.destroy(), self.nodeButtonPopUp()])
        nodeButton.grid(row=1, column=0)

        locButton = tk.Button(popupFrame, bg="indian red", fg="snow", text="X-Y Coordinates", command= lambda: [popupWin.destroy(), self.locationButtonPopUp()])
        locButton.grid(row=2, column=0)

        self.mainWin.wait_window(popupWin)

    def nodeButtonPopUp(self):
        """
        The pop up window that shows up if node entry is chosen
        :return:
        """
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input start node)")
        popupWin.geometry("+550+400")
        self.userInputStartNode.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popupWin, bg="indian red", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        nodeInfo = tk.Label(popupFrame, bg="indian red", fg="snow",
                           text="Enter start node (99 to quit): ")
        nodeInfo.grid(row=0, column=0)

        nodeInput = tk.Entry(popupFrame, textvariable=self.userInputStartNode)
        nodeInput.grid(row=1, column=0)

        yawInfo = tk.Label(popupFrame, bg="indian red", fg="snow", text="Enter start yaw (99 to quit): ")
        yawInfo.grid(row=2, column=0)

        yawNodeInput = tk.Entry(popupFrame, textvariable=self.userInputStartYaw)
        yawNodeInput.grid(row=3, column=0)

        confirmButton = tk.Button(popupFrame, bg="indian red", fg="snow", text="Confirm", command=popupWin.destroy)
        confirmButton.grid(row=4, column=0)

        self.mainWin.wait_window(popupWin)


    def locationButtonPopUp(self):
        """
        The pop up window that shows up when the coordinate button is chosen
        :return:
        """
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input start Coordinates)")
        popupWin.geometry("+550+400")
        self.userInputStartX.set("")
        self.userInputStartY.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popupWin, bg="indian red", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        xInfo = tk.Label(popupFrame, bg="indian red", fg="snow",
                           text="Enter x coordinate (99 to quit): ")
        xInfo.grid(row=0, column=0)

        xInput = tk.Entry(popupFrame, textvariable=self.userInputStartX)
        xInput.grid(row=1, column=0)

        yInfo = tk.Label(popupFrame, bg="indian red", fg="snow",
                           text="Enter y coordinate (99 to quit): ")
        yInfo.grid(row=0, column=0)

        yInput = tk.Entry(popupFrame, textvariable=self.userInputStartY)
        yInput.grid(row=1, column=0)

        yawInfo = tk.Label(popupFrame, bg="indian red", fg="snow", text="Enter start yaw (99 to quit): ")
        yawInfo.grid(row=2, column=0)

        yawCoordInput = tk.Entry(popupFrame, textvariable=self.userInputStartYaw)
        yawCoordInput.grid(row=3, column=0)

        confirmButton = tk.Button(popupFrame, bg="indian red", fg="snow", text="Confirm", command=popupWin.destroy)
        confirmButton.grid(row=4, column=0)

        self.mainWin.wait_window(popupWin)

    def popupStart(self):
        """
        The popup window that asks the user to put in the starting position
        :return:
        """
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input start node/location (space in between x and y)")
        popupWin.geometry("+550+400")
        self.userInputStartLoc.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popupWin, bg="indian red", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        locInfo = tk.Label(popupFrame, bg="indian red", fg="snow", text="Enter start node index or location (99 to quit): ")
        locInfo.grid(row=0, column=0)

        locInput = tk.Entry(popupFrame, textvariable=self.userInputStartLoc)
        locInput.grid(row=1, column=0)

        yawInfo =  tk.Label(popupFrame, bg="indian red", fg="snow", text="Enter start yaw (99 to quit): ")
        yawInfo.grid(row=2, column=0)

        yawInput = tk.Entry(popupFrame, textvariable=self.userInputStartYaw)
        yawInput.grid(row=3, column=0)

        confirmButton = tk.Button(popupFrame, bg="indian red", fg="snow", text="Confirm", command=popupWin.destroy)
        confirmButton.grid(row=4, column=0)


        self.mainWin.wait_window(popupWin)

    def popupDest(self):
        """
        The popup window that askes the reader to type a destination node number.
        """
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input destination")
        popupWin.geometry("+550+400")
        self.userInputDest.set("")

        self.confirmClicked = False

        popupFrame = tk.Frame(popupWin, bg="indian red", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        info = tk.Label(popupFrame, bg="indian red", fg="snow", text = "Enter destination index (99 to quit): ")
        info.grid(row=0, column=0)

        input = tk.Entry(popupFrame, textvariable = self.userInputDest)
        input.grid(row=1, column=0)

        confirmButton = tk.Button(popupFrame,  bg="indian red", fg="snow", text = "Confirm", command= popupWin.destroy)
        confirmButton.grid(row=2, column=0)

        self.mainWin.wait_window(popupWin)

    def inputDes(self):
        return self.userInputDest.get()

    def inputStartLoc(self):
        return self.userInputStartLoc.get()

    def inputStartYaw(self):
        return self.userInputStartYaw.get()

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
        # Heading frames
        headFrame = tk.Frame(self.mainWin, bg="gray22", bd=2, relief=tk.GROOVE)
        headFrame.grid(row = 0, column = 0, columnspan=4)
        modeLab = tk.Label(headFrame, bg="gray22", text="Mode:", font="DroidSans 24", width=8, height = 1)
        modeLab.grid(row = 0, column = 0)
        self.currentMode = tk.Label(headFrame, bg="gray22", textvariable=self.mode, font="DroidSans 24 bold", width=17, height =1)
        self.currentMode.grid(row = 0, column = 1)
        navLab = tk.Label(headFrame, bg="gray22", text="Nav by:", font="DroidSans 24", width=8, height =1)
        navLab.grid(row = 0, column = 2)
        self.currentNav = tk.Label(headFrame, textvariable=self.navType, bg="gray22", font="DroidSans 24 bold", width=17, height =1)
        self.currentNav.grid(row = 0, column = 3)


    def setUpLocFrame(self):
        #Locations frames
        locationsFrame = tk.Frame(self.mainWin, bg="gray22", bd=2, relief=tk.GROOVE)
        locationsFrame.grid(row = 1, column = 0)

        #row heading
        odomLabel = tk.Label(locationsFrame, bg="cornflower blue", text="Odometry:", font="DroidSans 16", width=16)
        odomLabel.grid(row = 1, column = 0)
        lastKnownLabel = tk.Label(locationsFrame, bg="cornflower blue", text="Last Known Loc:", font="DroidSans 16", width=16)
        lastKnownLabel.grid(row = 2, column = 0)
        MCLLabel = tk.Label(locationsFrame, bg="cornflower blue", text="MCL Center:", font="DroidSans 16", width=16)
        MCLLabel.grid(row = 3, column = 0)
        ClosestPic = tk.Label(locationsFrame, bg="cornflower blue", text="Best Match:", font="DroidSans 16", width=16)
        ClosestPic.grid(row = 4, column = 0)
        secondPic = tk.Label(locationsFrame, bg="cornflower blue", text="Second Match:", font="DroidSans 16", width=16)
        secondPic.grid(row = 5, column = 0)
        thirdPic = tk.Label(locationsFrame, bg="cornflower blue", text="Third Match:", font="DroidSans 16", width=16)
        thirdPic.grid(row = 6, column = 0)

        #column heading
        xLabel = tk.Label(locationsFrame, bg="cornflower blue", text="x", font="DroidSans 16", width=6)
        xLabel.grid(row = 0, column = 1)
        yLabel = tk.Label(locationsFrame, bg="cornflower blue", text="y", font="DroidSans 16", width=6)
        yLabel.grid(row = 0, column = 2)
        hLabel = tk.Label(locationsFrame, bg="cornflower blue", text="h", font="DroidSans 16", width=6)
        hLabel.grid(row = 0, column = 3)
        confLabel = tk.Label(locationsFrame, bg="cornflower blue", text="conf", font="DroidSans 16", width=6)
        confLabel.grid(row = 0, column = 4)

        #value display, odom info

        odomX = tk.Label(locationsFrame, textvariable=self.odomList[0], bg="light blue", font="DroidSans 16", width=5)
        odomX.grid(row = 1, column = 1)
        odomY = tk.Label(locationsFrame, textvariable=self.odomList[1], bg="light blue", font="DroidSans 16", width=5)
        odomY.grid(row = 1, column = 2)
        odomH = tk.Label(locationsFrame, textvariable=self.odomList[2], bg="light blue", font="DroidSans 16", width=6)
        odomH.grid(row = 1, column = 3)
        odomConf = tk.Label(locationsFrame, textvariable=self.odomList[3], bg="light blue", font="DroidSans 16", width=5)
        odomConf.grid(row = 1, column = 4)

        #value display, last know loc info
        lastKnownX = tk.Label(locationsFrame, textvariable=self.lastKnownList[0], bg="light blue", font="DroidSans 16", width=5)
        lastKnownX.grid(row = 2, column = 1)
        lastKnownY = tk.Label(locationsFrame, textvariable=self.lastKnownList[1], bg="light blue", font="DroidSans 16", width=5)
        lastKnownY.grid(row = 2, column = 2)
        lastKnownH = tk.Label(locationsFrame, textvariable=self.lastKnownList[2], bg="light blue", font="DroidSans 16", width=6)
        lastKnownH.grid(row = 2, column = 3)
        lastKnownConf = tk.Label(locationsFrame, textvariable=self.lastKnownList[3], bg="light blue", font="DroidSans 16", width=5)
        lastKnownConf.grid(row = 2, column = 4)

        #value display, MCl Center of Mass
        MCLX = tk.Label(locationsFrame, textvariable=self.MCLList[0], bg="light blue", font="DroidSans 16", width=5)
        MCLX.grid(row = 3, column = 1)
        MCLY = tk.Label(locationsFrame, textvariable=self.MCLList[1], bg="light blue", font="DroidSans 16", width=5)
        MCLY.grid(row = 3, column = 2)
        MCLH = tk.Label(locationsFrame, textvariable=self.MCLList[2], bg="light blue", font="DroidSans 16", width=6)
        MCLH.grid(row = 3, column = 3)
        self.MCLConf = tk.Label(locationsFrame, textvariable=self.MCLList[3], bg="light blue", font="DroidSans 16", width=5)
        self.MCLConf.grid(row = 3, column = 4)

        #value display, best pic
        bestPicX = tk.Label(locationsFrame, textvariable=self.bestPicList[0], bg="light blue", font="DroidSans 16", width=5)
        bestPicX.grid(row = 4, column = 1)
        bestPicY = tk.Label(locationsFrame, textvariable=self.bestPicList[1], bg="light blue", font="DroidSans 16", width=5)
        bestPicY.grid(row = 4, column = 2)
        bestPicH = tk.Label(locationsFrame, textvariable=self.bestPicList[2], bg="light blue", font="DroidSans 16", width=6)
        bestPicH.grid(row = 4, column = 3)
        self.bestPicConf = tk.Label(locationsFrame, textvariable=self.bestPicList[3], bg="light blue", font="DroidSans 16", width=5)
        self.bestPicConf.grid(row = 4, column = 4)

        #value display, second closest pic
        secondPicX = tk.Label(locationsFrame, textvariable=self.secondPicList[0], bg="light blue", font="DroidSans 16", width=5)
        secondPicX.grid(row = 5, column = 1)
        secondPicY = tk.Label(locationsFrame, textvariable=self.secondPicList[1], bg="light blue", font="DroidSans 16", width=5)
        secondPicY.grid(row = 5, column = 2)
        secondPicH = tk.Label(locationsFrame, textvariable=self.secondPicList[2], bg="light blue", font="DroidSans 16", width=6)
        secondPicH.grid(row = 5, column = 3)
        secondPicConf = tk.Label(locationsFrame, textvariable=self.secondPicList[3], bg="light blue", font="DroidSans 16", width=5)
        secondPicConf.grid(row = 5, column = 4)

        #value display, third closest pic
        thirdPicX = tk.Label(locationsFrame, textvariable=self.thirdPicList[0], bg="light blue", font="DroidSans 16", width=5)
        thirdPicX.grid(row = 6, column = 1)
        thirdPicY = tk.Label(locationsFrame, textvariable=self.thirdPicList[1], bg="light blue", font="DroidSans 16", width=5)
        thirdPicY.grid(row = 6, column = 2)
        thirdPicH = tk.Label(locationsFrame, textvariable=self.thirdPicList[2], bg="light blue", font="DroidSans 16", width=6)
        thirdPicH.grid(row = 6, column = 3)
        thirdPicConf = tk.Label(locationsFrame, textvariable=self.thirdPicList[3], bg="light blue", font="DroidSans 16", width=5)
        thirdPicConf.grid(row = 6, column = 4)



    def setUpMessages(self):
        # Messages frames
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
        # Coordinate check frames
        turnCheckFrame = tk.Frame(self.mainWin, bg="pink", bd=2, relief=tk.GROOVE)
        turnCheckFrame.grid(row = 2, column = 1)
        coordCheck= tk.Label(turnCheckFrame, bg="pale violet red", text="Turn Info", font="DroidSans 22 bold", width=25)
        coordCheck.grid(row = 0, column = 0, columnspan = 2)

        # display info, turning information
        turn = tk.Label(turnCheckFrame, textvariable = self.turnState, bg="pink", font = "DroidSans 15", width =30)
        turn.grid(row = 1, column = 0, columnspan = 2, pady = 5)

        #display info, currHead
        currHead1 = tk.Label(turnCheckFrame, text= "currHead =", bg = "pink", font = "DroidSans 15", width = 10)
        currHead1.grid(row = 2, column = 0, pady = 1)
        self.currHead2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[0], bg="pink", font = "DroidSans 15", width = 10)
        self.currHead2.grid(row = 2, column = 1, pady = 1)

        # display info, targetAngle
        targetAngle1 = tk.Label(turnCheckFrame, text="tAngle =", bg="pink", font="DroidSans 15", width=10)
        targetAngle1.grid(row = 3, column = 0, pady = 1)
        self.targetAngle2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[1], bg="pink", font="DroidSans 15", width=10)
        self.targetAngle2.grid(row = 3, column = 1, pady = 1)

        # display info, AngleToTurn
        angleToTurn1= tk.Label(turnCheckFrame, text="turnAngle =", bg="pink", font="DroidSans 15", width=10)
        angleToTurn1.grid(row = 4, column = 0, pady = 1)
        self.angleToTurn2 = tk.Label(turnCheckFrame, textvariable=self.turnInfo[2], bg="pink", font="DroidSans 15", width=10)
        self.angleToTurn2.grid(row =4, column = 1, pady = 1)


    def setUpImgMatch(self):
        # Image matching frames
        imageMatchFrame = tk.Frame(self.mainWin, bg="plum1", bd=2, relief=tk.GROOVE)
        imageMatchFrame.grid(row = 1, column = 1)
        imageMatchInfo = tk.Label(imageMatchFrame, bg="plum3", text="Image Matching Info", font="DroidSans 22 bold", width=25)
        imageMatchInfo.grid(row = 0, column = 0, columnspan = 2)

        #display info, closest node
        closestNode1 = tk.Label(imageMatchFrame, text="Closest Node: ", bg="plum1", font="DroidSans 15", width=13)
        closestNode1.grid(row = 1, column = 0)
        closestNode2 = tk.Label(imageMatchFrame, textvariable=self.cNode, bg="plum1", font="DroidSans 15", width=12)
        closestNode2.grid(row = 1, column = 1)

        #display info, next node
        nextNode1 = tk.Label(imageMatchFrame, text="Next Node: ", bg="plum1", font="DroidSans 15", width=13)
        nextNode1.grid(row=2, column=0)
        nextNode2 = tk.Label(imageMatchFrame, textvariable=self.nextNode, bg="plum1", font="DroidSans 15", width=12)
        nextNode2.grid(row=2, column=1)

        # display info, tDist
        tDist1 = tk.Label(imageMatchFrame, text="Target Dist:", bg="plum1", font="DroidSans 15", width=10)
        tDist1.grid(row=3, column=0)
        tDist2 = tk.Label(imageMatchFrame, textvariable=self.tDist, bg="plum1", font="DroidSans 15", width=10)
        tDist2.grid(row=3, column=1)

        #display info, search radius
        searchRadius1 = tk.Label(imageMatchFrame, text="Search Radius: ", bg="plum1", font="DroidSans 15", width=13)
        searchRadius1.grid(row = 4, column = 0)
        searchRadius2 = tk.Label(imageMatchFrame, textvariable=self.radius, bg="plum1", font="DroidSans 15", width=12)
        searchRadius2.grid(row = 4, column = 1)


        #display info, match status
        matchStatus1 = tk.Label(imageMatchFrame, text="Match Status: ", bg="plum1", font="DroidSans 15", width=13)
        matchStatus1.grid(row = 5, column = 0, pady = 1)
        matchStatus2 = tk.Label(imageMatchFrame, textvariable=self.matchStatusInfo, bg="plum1", font="DroidSans 15", width=25, justify="right")
        matchStatus2.grid(row = 6, column = 0, columnspan=2)


    def updateOdomList(self,loc):
        for i in range(len(self.odomList)):
            self.odomList[i].set('%.2f'%loc[i])

    def updateLastKnownList(self,loc):
        for i in range(len(self.lastKnownList)):
            self.lastKnownList[i].set('%.2f'%loc[i])

    def updateMCLList(self,loc):
        if self.navType.get() == "MCL":
            self.MCLConf.configure(bg="cornflower blue")
        else:
            self.MCLConf.configure(bg="light blue")

        for i in range(len(self.MCLList)):
            self.MCLList[i].set('%.2f'%loc[i])

    def updatePicLocs(self,loc1, loc2, loc3):
        if self.navType.get() == "Images":
            self.bestPicConf.configure(bg="cornflower blue")
        else:
            self.bestPicConf.configure(bg="light blue")

        for i in range(len(loc1)):
            self.bestPicList[i].set('%.2f'%loc1[i])
            self.secondPicList[i].set('%.2f'%loc2[i])
            self.thirdPicList[i].set('%.2f'%loc3[i])

    def updatePicConf(self,scores):
        self.bestPicList[3].set('%.2f'%scores[0])
        self.secondPicList[3].set('%.2f'%scores[1])
        self.thirdPicList[3].set('%.2f'%scores[2])

    def updateMessageText(self,text):
        if self.oddMessColor:
            self.messages.configure(bg="light goldenrod yellow")
            self.oddMessColor = False
        else:
            self.messages.configure(bg="light goldenrod")
            self.oddMessColor = True
        self.messages.insert('1.0',text+"\n")

    def updateTurnState(self,statement):
        self.turnState.set(statement)

    def updateTurnInfo(self,turnData):
        self.currHead2.configure(bg="pink")
        self.targetAngle2.configure(bg="pink")
        self.angleToTurn2.configure(bg="pink")
        for i in range(len(turnData)):
            self.turnInfo[i].set('%.2f'%turnData[i])

    def endTurn(self):
        s= "Was " + self.turnState.get()
        self.updateTurnState(s)
        self.currHead2.configure(bg="MistyRose2")
        self.targetAngle2.configure(bg="MistyRose2")
        self.angleToTurn2.configure(bg="MistyRose2")


    def updateTDist(self,dist):
        self.tDist.set('%.2f'%dist)

    def updateCNode(self,closestNode):
        self.cNode.set(closestNode)

    def updateNextNode(self,node):
        self.nextNode.set(node)

    def updateRadius(self,radius):
        self.radius.set('%.2f'%radius)

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

if __name__ == '__main__':
    gui = SeekerGUI()
    gui.mainWin.mainloop()

