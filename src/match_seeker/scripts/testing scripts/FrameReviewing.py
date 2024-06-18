import cv2
import os

def FrameReview():
    """ This function takes in a dictionary, with frame folders as keys and txt files as values, and iterates through each folder.
    For each frame displayed, it calls TextFileIteration to find the associated line in the txt file. When finished with the txt file,
    it closes the newly written in txt file and moves on to the next image. """

    folderPath = "../../res/classifier2022Data/DATA/FrameData/"
    folderList = sorted(os.listdir(folderPath))

    associatedTxtDict = {"20220705-1446frames": "Data-Jul05Tue-160449.txt"}  # TODO: add entries for the remaining folders that need to be reviewed

    for folder in folderList:
        # prints file name and asks if user wants to iterate through the folder. If no, pass.
        iterate = input("text file:  " + folder + "       Would you enter this file? (y/n): ")
        if iterate.lower() == "n":
            pass
        else:
            if folder.endswith("frames"):
                print('folder: ' + folder)
                currentPath = folderPath + folder + "/"
                currentFolderList = sorted(os.listdir(currentPath))
                reviewFile = open(folder + "Reviewed.txt", "w")
                for file in currentFolderList:
                    if file.endswith(".jpg"):
                        image = cv2.imread(currentPath + file)
                        print('showing: ' + file)
                        cv2.imshow('next image', image)
                        cv2.waitKey(60)
                        if folder in associatedTxtDict.keys():
                            TextFileIteration(associatedTxtDict[folder], reviewFile)
                reviewFile.close()
    cv2.destroyAllWindows()


def TextFileIteration(txtFile, newFile):
    """ Takes in the text file associated with the current frame data folder that is being iterated through by FrameReview.
    When the image currently being looked at corresponds to the current line in the text file (which can be changed with user
    input), the function adds a new line to a new text file returns to FrameReview. """

    textFolder = "../../res/locdata2022/"

    currentTextPath = textFolder + txtFile
    if txtFile.endswith(".txt"):
        currIndex = 0
        with open(currentTextPath) as textFile:
            lines = textFile.readlines()        # potential issue: each entry ends with \n and begins with a line number -- does it need to be removed?? (might mess with writing the new file)
            while True:
                print(lines[currIndex])
                ch = input("\nEnter a letter: a = backwards a line     d = forwards a line      k = correct text for frame (move to next image) \n")
                ch = ch.lower()
                if ch == "a" and (currIndex - 1 >= 0):
                    currIndex -= 1
                elif ch == "d" and (currIndex + 1 <= (len(lines)-1)):
                    currIndex += 1
                elif ch == "k":
                    print('okay')
                    # assign frame to current line of txt file
                    newFile.write("put info here")      # TODO: fill this to have the correct info we want from txt file and frame name
                    return


FrameReview()
