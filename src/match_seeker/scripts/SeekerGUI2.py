import Tkinter as tk

mainWindow = tk.Tk()
# frame = tk.Frame(mainWindow, bg="gray22")
# frame.grid(row=0, column=0, columnspan=4)
# label1 = tk.Label(frame, bg="gray22", relief="solid", text="lala", width=1, height=8)
#
#
# mainWindow.mainloop()

headFrame = tk.Frame(mainWindow, bg="gray22", bd=2, relief=tk.GROOVE)
headFrame.grid(row = 0, column = 0, columnspan=4, rowspan=3)
modeLab = tk.Label(headFrame, bg="gray22", text="Mode:", font="MSSansSerif 14", width=6, height = 1)
modeLab.config(fg="red")
modeLab.grid(row = 0, column = 0)
currentMode = tk.Label(headFrame, bg="gray22", textvariable="x", font="MSSansSerif 14 bold", width=10, height =1)
currentMode.grid(row = 0, column = 1)
navLab = tk.Label(headFrame, bg="gray22", text="Nav by:", font="MSSansSerif 14", width=8, height =1)
navLab.config(fg="red")
navLab.grid(row = 1, column = 0)
currentNav = tk.Label(headFrame, textvariable="X", bg="gray22", font="MSSansSerif 14 bold", width=10, height =1)
currentNav.grid(row = 1, column = 1)
predictLab = tk.Label(headFrame, bg="gray22", text="Prediction:", font="MSSansSerif 14", width=10, height=1)
predictLab.config(fg="red")
predictLab.grid(row=2, column=0)

###############################################################################################################

locationsFrame = tk.Frame(mainWindow, bg="gray22", bd=2, relief=tk.GROOVE)
locationsFrame.grid(row = 4, column = 0)

#row heading
odomLabel = tk.Label(locationsFrame, bg="gray22", text="Odometry:", font="MSSansSerif 14", width=14)
odomLabel.config(fg="dark orange")
odomLabel.grid(row = 1, column = 0)
lastKnownLabel = tk.Label(locationsFrame, bg="gray22", text="Last Known Loc:", font="MSSansSerif 14", width=14)
lastKnownLabel.config(fg="dark orange")
lastKnownLabel.grid(row = 2, column = 0)
MCLLabel = tk.Label(locationsFrame, bg="gray22", text="MCL Center:", font="MSSansSerif 14", width=14)
MCLLabel.config(fg="dark orange")
MCLLabel.grid(row = 3, column = 0)
ClosestPic = tk.Label(locationsFrame, bg="gray22", text="Best Match:", font="MSSansSerif 14", width=14)
ClosestPic.config(fg="dark orange")
ClosestPic.grid(row = 4, column = 0)
secondPic = tk.Label(locationsFrame, bg="gray22", text="Second Match:", font="MSSansSerif 14", width=14)
secondPic.config(fg="dark orange")
secondPic.grid(row = 5, column = 0)
thirdPic = tk.Label(locationsFrame, bg="gray22", text="Third Match:", font="MSSansSerif 14", width=14)
thirdPic.config(fg="dark orange")
thirdPic.grid(row = 6, column = 0)

#column heading
xLabel = tk.Label(locationsFrame, bg="gray22", text="x", font="MSSansSerif 14", width=6)
xLabel.config(fg="dark orange")
xLabel.grid(row = 0, column = 1)
yLabel = tk.Label(locationsFrame, bg="gray22", text="y", font="MSSansSerif 14", width=6)
yLabel.config(fg="dark orange")
yLabel.grid(row = 0, column = 2)
hLabel = tk.Label(locationsFrame, bg="gray22", text="h", font="MSSansSerif 14", width=6)
hLabel.config(fg="dark orange")
hLabel.grid(row = 0, column = 3)
confLabel = tk.Label(locationsFrame, bg="gray22", text="conf", font="MSSansSerif 14", width=6)
confLabel.config(fg="dark orange")
confLabel.grid(row = 0, column = 4)

#value display, odom info

odomX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
odomX.config(fg="dark orange")
odomX.grid(row = 1, column = 1)
odomY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
odomY.config(fg="dark orange")
odomY.grid(row = 1, column = 2)
odomH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
odomH.config(fg="dark orange")
odomH.grid(row = 1, column = 3)
odomConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
odomConf.config(fg="dark orange")
odomConf.grid(row = 1, column = 4)

#value display, last know loc info
lastKnownX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
lastKnownX.config(fg="dark orange")
lastKnownX.grid(row = 2, column = 1)
lastKnownY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
lastKnownY.config(fg="dark orange")
lastKnownY.grid(row = 2, column = 2)
lastKnownH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
lastKnownH.config(fg="dark orange")
lastKnownH.grid(row = 2, column = 3)
lastKnownConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
lastKnownConf.config(fg="dark orange")
lastKnownConf.grid(row = 2, column = 4)

#value display, MCl Center of Mass
MCLX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
MCLX.config(fg="dark orange")
MCLX.grid(row = 3, column = 1)
MCLY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
MCLY.config(fg="dark orange")
MCLY.grid(row = 3, column = 2)
MCLH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
MCLH.config(fg="dark orange")
MCLH.grid(row = 3, column = 3)
MCLConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
MCLConf.config(fg="dark orange")
MCLConf.grid(row = 3, column = 4)

#value display, best pic
bestPicX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
bestPicX.config(fg="dark orange")
bestPicX.grid(row = 4, column = 1)
bestPicY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
bestPicY.config(fg="dark orange")
bestPicY.grid(row = 4, column = 2)
bestPicH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
bestPicH.config(fg="dark orange")
bestPicH.grid(row = 4, column = 3)
bestPicConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
bestPicConf.config(fg="dark orange")
bestPicConf.grid(row = 4, column = 4)

#value display, second closest pic
secondPicX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
secondPicX.config(fg="dark orange")
secondPicX.grid(row = 5, column = 1)
secondPicY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
secondPicY.config(fg="dark orange")
secondPicY.grid(row = 5, column = 2)
secondPicH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
secondPicH.config(fg="dark orange")
secondPicH.grid(row = 5, column = 3)
secondPicConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
secondPicConf.config(fg="dark orange")
secondPicConf.grid(row = 5, column = 4)

#value display, third closest pic
thirdPicX = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
thirdPicX.config(fg="dark orange")
thirdPicX.grid(row = 6, column = 1)
thirdPicY = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
thirdPicY.config(fg="dark orange")
thirdPicY.grid(row = 6, column = 2)
thirdPicH = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=6)
thirdPicH.config(fg="dark orange")
thirdPicH.grid(row = 6, column = 3)
thirdPicConf = tk.Label(locationsFrame, bg="gray22", font="MSSansSerif 14", width=5)
thirdPicConf.config(fg="dark orange")
thirdPicConf.grid(row = 6, column = 4)

#############################################################################################################

# Messages frame
messageFrame = tk.Frame(mainWindow, bg="gray22", bd = 2, relief=tk.GROOVE, width = 480, height = 160)
messageFrame.grid(row = 5, column = 0)
# messageFrame.grid_propagate(0)

messLabel = tk.Label(messageFrame, bg="gray22", text = "Messages:", font="MSSansSerif 14 bold", width=31)
messLabel.config(fg="yellow")
messLabel.grid(row = 0, column = 0, columnspan = 2)
messages = tk.Text(messageFrame, bg="gray22",  wrap = tk.WORD, font="DroidSans 15",
                                   width = 43, height = 5) # width=15 height = 5
#messages.insert('end',messageText)
messages.grid(row = 1, column = 0)

scrollbar = tk.Scrollbar(messageFrame)
messages.config(yscrollcommand=scrollbar.set)
scrollbar.grid(row=1, column=1, sticky = 'ns')
scrollbar.config(command=messages.yview)


mainWindow.mainloop()
