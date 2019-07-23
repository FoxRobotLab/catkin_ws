import Tkinter as tk
from matchPlanner import MatchPlanner
from turtleControl import TurtleBot

class SeekerGUI2():

    def __init__(self, matchPlannerObj, turtleObj):

        self.turtleBot = turtleObj
        self.matchPlanner = matchPlannerObj

        self.mainWindow = tk.Tk()
        self.mainWindow.title("Seeker GUI")
        self.mainWindow.geometry("+550+650")

        self.mode = tk.StringVar()
        self.mode.set("Navigating")
        self.navType = tk.StringVar()
        self.navType.set("CNN")
        self.prediction = tk.StringVar()
        self.prediction.set("No Prediction")

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
        # self.currHead2 = None
        # self.targetAngle2 = None
        # self.angleToTurn2 = None

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

        self.frame1 = tk.Frame(self.mainWindow, width=700, height=100)
        self.frame1.config(bg="gray22")
        self.frame1.grid(row=0, column=0)
        self.canvas = tk.Canvas(self.mainWindow, width=700, height=700)
        self.canvas.config(bg="gray22")
        self.canvas.grid(row=1, column=0)
        self.frame2 = tk.Frame(self.mainWindow, width=700, height=300)
        self.frame2.config(bg="gray22")
        self.frame2.grid(row=2, column=0)

        self.button1 = tk.Button(self.frame1)

        self.setUpHeading()
        self.setUpButtons()
        self.setUpOdomGrid()
        self.setUpImgMatch()
        self.setUpTurnInfo()
        self.setUpMessages()

        self.confirmClickedQuit = False
        self.confirmClickedMotors = False

    def nodeButtonPopUp(self):
        def destroy_callback(e):
            popUpWindow.destroy()
        popUpWindow = tk.Toplevel()
        popUpWindow.wm_title("Input start node)")
        popUpWindow.geometry("+550+400")
        self.userInputStartNode.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popUpWindow, bg="gray22", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        nodeInfo = tk.Label(popupFrame, bg="gray22", fg="snow", font="MSSansSerif 14",
                            text="Enter start node (-1 to quit): ")
        nodeInfo.grid(row=0, column=0)

        nodeInput = tk.Entry(popupFrame, textvariable=self.userInputStartNode)
        nodeInput.grid(row=1, column=0)
        nodeInput.focus()

        yawInfo = tk.Label(popupFrame, bg="gray22", fg="snow", font="MSSansSerif 14",text="Enter start yaw (-1 to quit): ")
        yawInfo.grid(row=2, column=0)

        yawNodeInput = tk.Entry(popupFrame, textvariable=self.userInputStartYaw)
        yawNodeInput.grid(row=3, column=0)

        confirmButton = tk.Button(popupFrame, bg="gray22", fg="snow", text="Confirm", command=popUpWindow.destroy)
        confirmButton.grid(row=4, column=0)
        popUpWindow.bind("<Return>", destroy_callback)
        popUpWindow.bind("<KP_Enter>", destroy_callback)

        self.mainWindow.wait_window(popUpWindow)

    def popupStart(self):
        def destroy_callback(e):
            print("DESTROYING")
            popupWin.destroy()
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input start node/location (space in between x and y)")
        popupWin.geometry("+550+400")
        self.userInputStartLoc.set("")
        self.userInputStartYaw.set("")
        self.confirmClickedStart = False

        popupFrame = tk.Frame(popupWin, bg="gray22", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        locInfo = tk.Label(popupFrame, bg="gray22", fg="snow",
                           text="Enter start cell (-1 to quit): ")
        locInfo.grid(row=0, column=0)

        locInput = tk.Entry(popupFrame, textvariable=self.userInputStartLoc)
        locInput.grid(row=1, column=0)
        locInput.focus()

        yawInfo = tk.Label(popupFrame, bg="gray22", fg="snow", text="Enter start heading (-1 to quit): ")
        yawInfo.grid(row=2, column=0)

        yawInput = tk.Entry(popupFrame, textvariable=self.userInputStartYaw)
        yawInput.grid(row=3, column=0)

        confirmButton = tk.Button(popupFrame, bg="gray22", fg="snow", text="Confirm", command=popupWin.destroy)
        confirmButton.grid(row=4, column=0)
        popupWin.bind("<Return>", destroy_callback)
        popupWin.bind("<KP_Enter>", destroy_callback)

        self.mainWindow.wait_window(popupWin)



    def popupDest(self):
        """
        The popup window that asks the reader to type a destination node number.
        """
        def destroy_callback(e):
            popupWin.destroy()
        popupWin = tk.Toplevel()
        popupWin.wm_title("Input destination")
        popupWin.geometry("+550+400")
        self.userInputDest.set("")

        self.confirmClicked = False

        popupFrame = tk.Frame(popupWin, bg="gray22", bd=2, relief=tk.GROOVE)
        popupFrame.grid(row=0, column=0)

        info = tk.Label(popupFrame, bg="gray22", fg="snow", text = "Enter destination cell (-1 to quit): ")
        info.grid(row=0, column=0)

        input = tk.Entry(popupFrame, textvariable = self.userInputDest)
        input.grid(row=1, column=0)
        input.focus()

        confirmButton = tk.Button(popupFrame,  bg="gray22", fg="snow", text = "Confirm", command= popupWin.destroy)
        confirmButton.grid(row=2, column=0)
        popupWin.bind("<Return>", destroy_callback)
        popupWin.bind("<KP_Enter>", destroy_callback)

        self.mainWindow.wait_window(popupWin)

    def inputDes(self):
        return self.userInputDest.get()

    def inputStartLoc(self):
        return self.userInputStartLoc.get()

    def inputStartYaw(self):
        return self.userInputStartYaw.get()

    def setupLocLists(self):
        """each list has x, y, heading, and confidence"""
        for list in [self.odomList, self.lastKnownList, self.MCLList, self.bestPicList, self.secondPicList, self.thirdPicList]:
            for i in range(5):
                var = tk.StringVar()
                var.set(0.0)
                list.append(var)

    def setupTurnList(self):
        """a turn statement, currHead, targetHeading, angleToTurn, tDist"""
        for i in range(3):
            var = tk.StringVar()
            var.set(0.0)
            self.turnInfo.append(var)

    def setUpHeading(self):

        modeLab = tk.Label(self.canvas, bg="gray22", text="Mode:", font="MSSansSerif 14", width=6, height=1)
        modeLab.config(fg="dark orange")
        self.canvas.create_window(40, 15, window=modeLab)
        self.currentMode = tk.Label(self.canvas, bg="gray22", textvariable=self.mode, font="MSSansSerif 14 italic", width=17,
                                    height=1)
        self.currentMode.config(fg="dark orange")
        self.canvas.create_window(200, 15, window=self.currentMode)
        navLab = tk.Label(self.canvas, bg="gray22", text="Nav by:", font="MSSansSerif 14", width=6, height=1)
        navLab.config(fg="dark orange")
        self.canvas.create_window(48, 45, window=navLab)
        self.currentNav = tk.Label(self.canvas, textvariable=self.navType, bg="gray22", font="MSSansSerif 14 italic",
                                   width=16, height=1)
        self.currentNav.config(fg="dark orange")
        self.canvas.create_window(184, 45, window=self.currentNav)
        # predictLab = tk.Label(self.canvas, bg="gray22", text="Prediction:", font="MSSansSerif 14", width=8, height=1)
        # predictLab.config(fg="dark orange")
        # self.canvas.create_window(61, 75, window=predictLab)
        # self.currentPred = tk.Label(self.canvas, textvariable=self.prediction, bg="gray22", font="MSSansSerif 14 italic",
        #                             width=14, height=1)
        # self.currentPred.config(fg="dark orange")
        # self.canvas.create_window(210, 75, window=self.currentPred)

    def setUpButtons(self):


        self.button1.config(bg="gray22", fg="red", text="Stop Motors", font="MSSansSerif 14", border=5, relief="raised",
                            command=self.toggleMotors)
        self.button1.grid(row=0, column=1)

        blankLabel1 = tk.Label(self.frame1, bg="gray22", width=8)
        blankLabel1.grid(row=0, column=2)

        button2 = tk.Button(self.frame1, text="Quit Program", bg="gray22", fg="red", font="MSSansSerif 14", border=5,
                            relief="raised", command=self.quitProgram)
        button2.grid(row=0, column=3)

        blankLabel2 = tk.Label(self.frame1, bg="gray22", width=20)
        blankLabel2.grid(row=0, column=0)

        blankLabel3 = tk.Label(self.frame1, bg="gray22", width=20)
        blankLabel3.grid(row=0, column=5)

    def setUpOdomGrid(self):

        odomLabel = tk.Label(self.canvas, bg="gray22", text="Odometry:", font="MSSansSerif 14", width=8)
        odomLabel.config(fg="gold")
        self.canvas.create_window(59, 144, window=odomLabel)

        lastKnownLabel = tk.Label(self.canvas, bg="gray22", text="Last Known Loc:", font="MSSansSerif 14", width=12)
        lastKnownLabel.config(fg="gold")
        self.canvas.create_window(84, 177, window=lastKnownLabel)

        MCLLabel = tk.Label(self.canvas, bg="gray22", text="MCL Center:", font="MSSansSerif 14", width=10)
        MCLLabel.config(fg="gold")
        self.canvas.create_window(66, 212, window=MCLLabel)

        ClosestPic = tk.Label(self.canvas, bg="gray22", text="Best Match:", font="MSSansSerif 14", width=9)
        ClosestPic.config(fg="gold")
        self.canvas.create_window(64, 247, window=ClosestPic)

        secondPic = tk.Label(self.canvas, bg="gray22", text="Second Match:", font="MSSansSerif 14", width=12)
        secondPic.config(fg="gold")
        self.canvas.create_window(76, 282, window=secondPic)

        thirdPic = tk.Label(self.canvas, bg="gray22", text="Third Match:", font="MSSansSerif 14", width=10)
        thirdPic.config(fg="gold")
        self.canvas.create_window(67, 319, window=thirdPic)

        xLabel = tk.Label(self.canvas, bg="gray22", text="x", font="MSSansSerif 14", width=2)
        xLabel.config(fg="gold")
        self.canvas.create_window(225, 110, window=xLabel)

        yLabel = tk.Label(self.canvas, bg="gray22", text="y", font="MSSansSerif 14", width=6)
        yLabel.config(fg="gold")
        self.canvas.create_window(325, 110, window=yLabel)

        hLabel = tk.Label(self.canvas, bg="gray22", text="h", font="MSSansSerif 14", width=6)
        hLabel.config(fg="gold")
        self.canvas.create_window(425, 110, window=hLabel)

        cellLabel = tk.Label(self.canvas, bg="gray22", text="cell", font="MSSansSerif 14", width=6)
        cellLabel.config(fg="gold")
        self.canvas.create_window(525, 110, window=cellLabel)

        confLabel = tk.Label(self.canvas, bg="gray22", text="conf", font="MSSansSerif 14", width=6)
        confLabel.config(fg="gold")
        self.canvas.create_window(625, 110, window=confLabel)

        self.canvas.create_line(180, 105, 180, 333, fill="white")
        self.canvas.create_line(10, 125, 675, 125, fill="white")
        self.canvas.create_line(10, 160, 675, 160, fill="white")
        self.canvas.create_line(10, 195, 675, 195, fill="white")
        self.canvas.create_line(10, 230, 675, 230, fill="white")
        self.canvas.create_line(10, 265, 675, 265, fill="white")
        self.canvas.create_line(10, 300, 675, 300, fill="white")

        self.canvas.create_line(275, 105, 275, 333, fill="white")
        self.canvas.create_line(375, 105, 375, 333, fill="white")
        self.canvas.create_line(475, 105, 475, 333, fill="white")
        self.canvas.create_line(575, 105, 575, 333, fill="white")

        odomX = tk.Label(self.canvas, textvariable=self.odomList[0], bg="gray22", font="MSSansSerif 14 bold", width=5)
        odomX.config(fg="gold")
        self.canvas.create_window(225, 144, window=odomX)
        odomY = tk.Label(self.canvas, textvariable=self.odomList[1], bg="gray22", font="MSSansSerif 14 bold", width=5)
        odomY.config(fg="gold")
        self.canvas.create_window(325, 144, window=odomY)
        odomYaw = tk.Label(self.canvas, textvariable=self.odomList[2], bg="gray22", font="MSSansSerif 14 bold", width=6)
        odomYaw.config(fg="gold")
        self.canvas.create_window(425, 144, window=odomYaw)

        odomCell = tk.Label(self.canvas, textvariable=self.odomList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        odomCell.config(fg="gold")
        self.canvas.create_window(525, 144, window=odomCell)

        odomConf = tk.Label(self.canvas, textvariable=self.odomList[3], bg="gray22", font="MSSansSerif 14 bold",
                            width=5)
        odomConf.config(fg="gold")
        self.canvas.create_window(625, 144, window=odomConf)

        lastKnownX = tk.Label(self.canvas, textvariable=self.lastKnownList[0], bg="gray22", font="MSSansSerif 14 bold",
                              width=5)
        lastKnownX.config(fg="gold")
        self.canvas.create_window(225, 177, window=lastKnownX)
        lastKnownY = tk.Label(self.canvas, textvariable=self.lastKnownList[1], bg="gray22", font="MSSansSerif 14 bold",
                              width=5)
        lastKnownY.config(fg="gold")
        self.canvas.create_window(325, 177, window=lastKnownY)
        lastKnownYaw = tk.Label(self.canvas, textvariable=self.lastKnownList[2], bg="gray22", font="MSSansSerif 14 bold",
                              width=6)
        lastKnownYaw.config(fg="gold")
        self.canvas.create_window(425, 177, window=lastKnownYaw)
        lastKnownConf = tk.Label(self.canvas, textvariable=self.lastKnownList[3], bg="gray22", font="MSSansSerif 14 bold",
                                 width=5)

        lastKnownCell = tk.Label(self.canvas, textvariable=self.lastKnownList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        lastKnownCell.config(fg="gold")
        self.canvas.create_window(525, 177, window=lastKnownCell)

        lastKnownConf.config(fg="gold")
        self.canvas.create_window(625, 177, window=lastKnownConf)

        MCLX = tk.Label(self.canvas, textvariable=self.MCLList[0], bg="gray22", font="MSSansSerif 14 bold", width=5)
        MCLX.config(fg="gold")
        self.canvas.create_window(225, 212, window=MCLX)
        MCLY = tk.Label(self.canvas, textvariable=self.MCLList[1], bg="gray22", font="MSSansSerif 14 bold", width=5)
        MCLY.config(fg="gold")
        self.canvas.create_window(325, 212, window=MCLY)
        MCLYaw = tk.Label(self.canvas, textvariable=self.MCLList[2], bg="gray22", font="MSSansSerif 14 bold", width=6)
        MCLYaw.config(fg="gold")
        self.canvas.create_window(425, 212, window=MCLYaw)
        self.MCLConf = tk.Label(self.canvas, textvariable=self.MCLList[3], bg="gray22", font="MSSansSerif 14 bold",
                                width=5)

        MCLCell = tk.Label(self.canvas, textvariable=self.MCLList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        MCLCell.config(fg="gold")
        self.canvas.create_window(525, 212, window=MCLCell)

        self.MCLConf.config(fg="gold")
        self.canvas.create_window(625, 212, window=self.MCLConf)

        bestPicX = tk.Label(self.canvas, textvariable=self.bestPicList[0], bg="gray22", font="MSSansSerif 14 bold",
                            width=5)
        bestPicX.config(fg="gold")
        self.canvas.create_window(225, 247, window=bestPicX)
        bestPicY = tk.Label(self.canvas, textvariable=self.bestPicList[1], bg="gray22", font="MSSansSerif 14 bold",
                            width=5)
        bestPicY.config(fg="gold")
        self.canvas.create_window(325, 247, window=bestPicY)
        bestPicYaw = tk.Label(self.canvas, textvariable=self.bestPicList[2], bg="gray22", font="MSSansSerif 14 bold",
                            width=6)
        bestPicYaw.config(fg="gold")
        self.canvas.create_window(425, 247, window=bestPicYaw)
        self.bestPicConf = tk.Label(self.canvas, textvariable=self.bestPicList[3], bg="gray22",
                                    font="MSSansSerif 14 bold", width=5)

        bestCell = tk.Label(self.canvas, textvariable=self.bestPicList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        bestCell.config(fg="gold")
        self.canvas.create_window(525, 247, window=bestCell)

        self.bestPicConf.config(fg="gold")
        self.canvas.create_window(625, 247, window=self.bestPicConf)

        secondPicX = tk.Label(self.canvas, textvariable=self.secondPicList[0], bg="gray22", font="MSSansSerif 14 bold",
                              width=5)
        secondPicX.config(fg="gold")
        self.canvas.create_window(225, 282, window=secondPicX)
        secondPicY = tk.Label(self.canvas, textvariable=self.secondPicList[1], bg="gray22", font="MSSansSerif 14 bold",
                              width=5)
        secondPicY.config(fg="gold")
        self.canvas.create_window(325, 282, window=secondPicY)
        secondPicYaw = tk.Label(self.canvas, textvariable=self.secondPicList[2], bg="gray22", font="MSSansSerif 14 bold",
                              width=6)
        secondPicYaw.config(fg="gold")
        self.canvas.create_window(425, 282, window=secondPicYaw)
        secondPicConf = tk.Label(self.canvas, textvariable=self.secondPicList[3], bg="gray22",
                                 font="MSSansSerif 14 bold", width=5)

        secondCell = tk.Label(self.canvas, textvariable=self.secondPicList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        secondCell.config(fg="gold")
        self.canvas.create_window(525, 282, window=secondCell)

        secondPicConf.config(fg="gold")
        self.canvas.create_window(625, 282, window=secondPicConf)

        thirdPicX = tk.Label(self.canvas, textvariable=self.thirdPicList[0], bg="gray22", font="MSSansSerif 14 bold",
                             width=5)
        thirdPicX.config(fg="gold")
        self.canvas.create_window(225, 319, window=thirdPicX)
        thirdPicY = tk.Label(self.canvas, textvariable=self.thirdPicList[1], bg="gray22", font="MSSansSerif 14 bold",
                             width=5)
        thirdPicY.config(fg="gold")
        self.canvas.create_window(325, 319, window=thirdPicY)
        thirdPicYaw = tk.Label(self.canvas, textvariable=self.thirdPicList[2], bg="gray22", font="MSSansSerif 14 bold",
                             width=6)
        thirdPicYaw.config(fg="gold")
        self.canvas.create_window(425, 319, window=thirdPicYaw)
        thirdPicConf = tk.Label(self.canvas, textvariable=self.thirdPicList[3], bg="gray22", font="MSSansSerif 14 bold",
                                width=5)

        thirdCell = tk.Label(self.canvas, textvariable=self.thirdPicList[4], bg="gray22", font="MSSansSerif 14 bold", width=6)
        thirdCell.config(fg="gold")
        self.canvas.create_window(525, 319, window=thirdCell)

        thirdPicConf.config(fg="gold")
        self.canvas.create_window(625, 319, window=thirdPicConf)

    def setUpImgMatch(self):

        imageTitle = tk.Label(self.canvas, bg="gray22", text="Path Planning", font="MSSansSerif 14 bold")
        imageTitle.config(fg="green2")
        self.canvas.create_window(82, 395, window=imageTitle)

        # Didn't make sense in context of current system
        # closetLab = tk.Label(self.canvas, bg="gray22", text="Closest Node:", font="MSSansSerif 14", width=10)
        # closetLab.config(fg="green2")
        # self.canvas.create_window(69, 395, window=closetLab)
        # closestInfoLab = tk.Label(self.canvas, textvariable=self.cNode, bg="gray22", font="MSSansSerif 14 italic",
        #                           width=12)
        # closestInfoLab.config(fg="green2")
        # self.canvas.create_window(250, 395, window=closestInfoLab)

        nextLabel = tk.Label(self.canvas, bg="gray22", text="Next Node:", font="MSSansSerif 14", width=8)
        nextLabel.config(fg="green2")
        self.canvas.create_window(61, 425, window=nextLabel)
        nextInfoLabel = tk.Label(self.canvas, textvariable=self.nextNode, bg="gray22", font="MSSansSerif 14 italic",
                                 width=12)
        nextInfoLabel.config(fg="green2")
        self.canvas.create_window(247, 425, window=nextInfoLabel)

        targetLabel = tk.Label(self.canvas, bg="gray22", text="Target Dist:", font="MSSansSerif 14", width=10)
        targetLabel.config(fg="green2")
        self.canvas.create_window(65, 455, window=targetLabel)
        targetInfoLabel = tk.Label(self.canvas, textvariable=self.tDist, bg="gray22", font="MSSansSerif 14 italic",
                                   width=10)
        targetInfoLabel.config(fg="green2")
        self.canvas.create_window(250, 455, window=targetInfoLabel)

        # Didn't make sense in context of current system
        # searchLabel = tk.Label(self.canvas, bg="gray22", text="Search Radius:", font="MSSansSerif 14", width=12)
        # searchLabel.config(fg="green2")
        # self.canvas.create_window(79, 485, window=searchLabel)
        # searchInfoLabel = tk.Label(self.canvas, textvariable=self.radius, bg="gray22", font="MSSansSerif 14 italic",
        #                            width=12)
        # searchInfoLabel.config(fg="green2")
        # self.canvas.create_window(250, 485, window=searchInfoLabel)
        #
        # matchLabel = tk.Label(self.canvas, bg="gray22", text="Match Status:", font="MSSansSerif 14", width=10)
        # matchLabel.config(fg="green2")
        # self.canvas.create_window(71, 515, window=matchLabel)
        # matchInfoLabel = tk.Label(self.canvas, textvariable=self.matchStatusInfo, bg="gray22", font="MSSansSerif 14 italic",
        #                         width=25, justify="right")
        # matchInfoLabel.config(fg="green2")
        # self.canvas.create_window(360, 515, window=matchInfoLabel)

    def setUpTurnInfo(self):

        turnInfo = tk.Label(self.canvas, bg="gray22", text="Most Recent Turn", font="MSSansSerif 14 bold")
        turnInfo.config(fg="cyan")
        self.canvas.create_window(100, 555, window=turnInfo)

        turnStatus = tk.Label(self.canvas, textvariable=self.turnState, bg="gray22", font="MSSansSerif 14 italic", width=30)
        turnStatus.config(fg="cyan")
        self.canvas.create_window(325, 575, window=turnStatus)

        currLabel = tk.Label(self.canvas, bg="gray22", text="currHead:", font="MSSansSerif 14", width=8)
        currLabel.config(fg="cyan")
        self.canvas.create_window(56, 605, window=currLabel)
        self.currInfoLabel = tk.Label(self.canvas, textvariable=self.turnInfo[0], bg="gray22", font="MSSansSerif 14 italic",
                                  width=10)
        self.currInfoLabel.config(fg="cyan")
        self.canvas.create_window(250, 605, window=self.currInfoLabel)

        targetLabel = tk.Label(self.canvas, bg="gray22", text="Target Angle:", font="MSSansSerif 14", width=10)
        targetLabel.config(fg="cyan")
        self.canvas.create_window(72, 635, window=targetLabel)
        self.targetInfoLabel = tk.Label(self.canvas, textvariable=self.turnInfo[1], bg="gray22", font="MSSansSerif 14 italic",
                                     width=10)
        self.targetInfoLabel.config(fg="cyan")
        self.canvas.create_window(249, 635, window=self.targetInfoLabel)

        turnLabel = tk.Label(self.canvas, bg="gray22", text="turnAngle:", font="MSSansSerif 14", width=8)
        turnLabel.config(fg="cyan")
        self.canvas.create_window(60, 665, window=turnLabel)
        self.turnInfoLabel = tk.Label(self.canvas, textvariable=self.turnInfo[2], bg="gray22", font="MSSansSerif 14 italic",
                                     width=10)
        self.turnInfoLabel.config(fg="cyan")
        self.canvas.create_window(250, 665, window=self.turnInfoLabel)

    def setUpMessages(self):

        blankLabel1 = tk.Label(self.frame2, bg="gray22", width=11)
        blankLabel1.grid(row=0, column=0)

        blankLabel2 = tk.Label(self.frame2, bg="gray22", width=10)
        blankLabel2.grid(row=0, column=2)

        messLabel = tk.Label(self.frame2, bg="gray22", text="Messages:", font="MSSansSerif 14 bold", width=31)
        messLabel.config(fg="purple1")
        messLabel.grid(row=0, column=1, columnspan=3)
        self.messages = tk.Text(self.frame2, bg="gray22", wrap=tk.WORD, font="MSSansSerif 14",
                                width=43, height=5)  # width=15 height = 5
        self.messages.config(fg="purple1")
        self.messages.insert('end', self.messageText)
        self.messages.grid(row=1, column=1)

        scrollbar = tk.Scrollbar(self.frame2)
        self.messages.config(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=1, column=2, sticky='ns')
        scrollbar.config(command=self.messages.yview)

    def updateOdomList(self,loc):
        for i in range(len(self.odomList)-1):
            self.odomList[i].set('%.2f'%loc[i])
        self.odomList[4].set(self.locToCell(self.odomList))
        self.odomList[2].set(positive_heading(self.odomList[2]))


    def updateLastKnownList(self,loc):
        for i in range(len(self.lastKnownList)-1):
            self.lastKnownList[i].set('%.2f'%loc[i])
        self.lastKnownList[4].set(self.locToCell(self.lastKnownList))
        self.lastKnownList[2].set(positive_heading(self.lastKnownList[2]))

    def updateMCLList(self,loc):
        if self.navType.get() == "MCL":
            self.MCLConf.configure(bg="cornflower blue")
        else:
            self.MCLConf.configure(bg="light blue")

        for i in range(len(self.MCLList)-1):
            self.MCLList[i].set('%.2f'%loc[i])

        self.MCLList[4].set(self.locToCell(self.MCLList))
        self.MCLList[2].set(positive_heading(self.MCLList[2]))

    def updatePicLocs(self,loc1, loc2, loc3):

        # If the network prediction confidence is less than 20, ignore prediction
        if loc2 == loc1:
            loc2 = [-1,-1,-1]
            loc3 = [-1,-1,-1]
        elif loc3 == loc1:
            loc3 = [-1,-1,-1]
        if self.navType.get() == "CNN":
            self.bestPicConf.configure(bg="cornflower blue")
        else:
            self.bestPicConf.configure(bg="light blue")

        for i in range(len(loc1)):
            self.bestPicList[i].set('%.2f'%loc1[i])
            self.secondPicList[i].set('%.2f'%loc2[i])
            self.thirdPicList[i].set('%.2f'%loc3[i])

        self.bestPicList[4].set(self.locToCell(self.bestPicList))
        self.secondPicList[4].set(self.locToCell(self.secondPicList))
        self.thirdPicList[4].set(self.locToCell(self.thirdPicList))

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
        # self.currHead2.configure(bg="pink")
        # self.targetAngle2.configure(bg="pink")
        # self.angleToTurn2.configure(bg="pink")
        for i in range(len(turnData)):
            if i <2:
                self.turnInfo[i].set('%.2f'%positive_heading(turnData[i]))
            else:
                self.turnInfo[i].set('%.2f'%turnData[i])

    def endTurn(self):
        s= "Was " + self.turnState.get()
        self.updateTurnState(s)
        # self.currHead2.configure(bg="MistyRose2")
        # self.targetAngle2.configure(bg="MistyRose2")
        # self.angleToTurn2.configure(bg="MistyRose2")

    def toggleMotors(self):
        if self.button1.config('text')[-1] == 'Stop Motors':
            self.turtleBot.pauseMovement()
            #self.matchPlanner.brain.pause()
            self.button1.config(text="Run Motors")
        else:
            self.turtleBot.unpauseMovement()
            #self.matchPlanner.brain.unpause()
            self.button1.config(text="Stop Motors")

    def quitProgram(self):
        self.matchPlanner.shutdown()

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

    def tilePrediction(self):
        self.prediction.set("Tile")

    def carpetPrediction(self):
        self.prediction.set("Carpet")

    def stop(self):
        self.mainWindow.destroy()

    def update(self):
        try:
            self.mainWindow.update_idletasks()
            self.mainWindow.update()
        except tk.TclError:
            pass

    def locToCell(self,locList):
        x = float(locList[0].get())
        y = float(locList[1].get())
        h = float(locList[2].get())
        cell = self.matchPlanner.olinMap.convertLocToCell((x,y,h))
        return str(cell)

def positive_heading(heading):
    if type(heading) is float or type(heading) is int:
        if heading < 0:
            return heading + 360
        else:
            return heading
    else:
        if float(heading.get()) < 0:
            heading.set(str(float(heading.get())+360))
            return "%.2f" % float(heading.get())
        return heading.get()

if __name__ == '__main__':
    gui = SeekerGUI2()
    gui.mainWindow.mainloop()
