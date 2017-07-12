
import Tkinter as tk



class SeekerGUI():


    def __init__(self):
        """Sets up all the GUI elements"""


        self.mainWin = tk.Tk()
        self.mainWin.title("Match Seeker")

        # Heading frame
        headingFrame = tk.Frame(self.mainWin, bg="light green", bd=2, relief=tk.GROOVE)
        headingFrame.grid(row = 0, column = 0, columnspan=2)
        modeLab = tk.Label(headingFrame, text="Mode", font="DroidSans 24")
        modeLab.grid(row = 0, column = 0)
        self.currentMode = tk.Label(headingFrame, text="Navigating", font="DroidSans 24 bold", width=10,
                                    bd=1, relief=tk.RAISED)
        self.currentMode.grid(row = 0, column = 1)
        navLab = tk.Label(headingFrame, text="Nav by:", font="DroidSans 24")
        navLab.grid(row = 1, column = 0)
        self.currentNav = tk.Label(headingFrame, text="ImageMatch", font="DroidSans 24 bold", width=10,
                                    bd=1, relief=tk.RAISED)
        self.currentNav.grid(row = 1, column = 1)


        #Locations frame
        locationsFrame = tk.Frame(self.mainWin, bg="light blue", bd=2, relief=tk.GROOVE)
        locationsFrame.grid(row = 1, column = 0)
        odomLabel = tk.Label(locationsFrame, text="Odometry Location:", font="DroidSans 16", width=20)
        odomLabel.grid(row = 1, column = 0)
        xLabel = tk.Label(locationsFrame, text="X", font="DroidSans 16", width=5)
        xLabel.grid(row = 0, column = 1)
        yLabel = tk.Label(locationsFrame, text="Y", font="DroidSans 16", width=5)
        yLabel.grid(row = 0, column = 2)
        hLabel = tk.Label(locationsFrame, text="H", font="DroidSans 16", width=5)
        hLabel.grid(row = 0, column = 3)
        confLabel = tk.Label(locationsFrame, text="Conf", font="DroidSans 16", width=5)
        confLabel.grid(row = 0, column = 4)


        # Messages frame
        messageFrame = tk.Frame(self.mainWin, bg="light yellow", bd = 2, relief=tk.GROOVE)
        messageFrame.grid(row = 2, column = 0)
        messLabel = tk.Label(messageFrame, text = "Messages:", font="DroidSans 22 bold", width=15)
        messLabel.grid(row = 0, column = 0)
        self.messages = tk.Label(messageFrame, text = "", font="DroidSans 22")
        self.messages.grid(row = 1, column = 0)

        # Coordinate check frame
        turnCheckFrame = tk.Frame(self.mainWin, bg="light pink", bd=2, relief=tk.GROOVE)
        turnCheckFrame.grid(row = 1, column = 1)
        turnLab = tk.Label(turnCheckFrame, text="Coordinate Check", font="DroidSans 16", width=25)
        turnLab.grid(row = 0, column = 0)

        # Image matching frame
        imageMatchFrame = tk.Frame(self.mainWin, bg="violet", bd=2, relief=tk.GROOVE)
        imageMatchFrame.grid(row = 2, column = 1)
        turnLab = tk.Label(imageMatchFrame, text="Image Matching Info", font="DroidSans 16", width=25)
        turnLab.grid(row = 0, column = 0)

    def go(self):
        """Set the GUI running"""
        self.mainWin.mainloop()



if __name__ == '__main__':
    gui = SeekerGUI()
    gui.go()
