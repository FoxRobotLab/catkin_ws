from ImageDataset import ImageDataset
import cv2
from random import randint, choice
from tqdm import tqdm
from OutputLogger import OutputLogger
import math

basePath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/"
#
turtleBase = 'kobuki' # os.environ["TURTLEBOT_BASE"]
#
if turtleBase == "create":
    imageDirectory = "res/create060717/"
    locData = "res/locations/SUSANcreate0607.txt"
elif turtleBase == "kobuki":
    # imageDirectory = "res/kobuki0615/"
    imageDirectory = "res/allFrames060418/"
    # locData = "res/locations/Susankobuki0615.txt"
    locData = "res/locdata/allLocs060418.txt"

def getAccuracy():
    num_eval = 1500
    correctCells = 0
    correctHeadings = 0
    with open('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/MASTER_CELL_LOC_FRAME_IDENTIFIER.txt','r') as masterlist:
        frame_heading = {}
        dataset = ImageDataset(OutputLogger(False,False),None)
        dataset.setupData(basePath + imageDirectory, basePath + locData, "frame", "jpg")
        for line in masterlist.readlines():
            # line = line.strip()
            line = line.split()
            frame = "{0:04d}".format(int(line[0]))
            x = float(line[2])
            y = float(line[3])
            h = float(line[4])
            frame_heading[frame] = [x, y, h]
        correctC = 0
        correctH = 0
        frameKeys = frame_heading.keys()
        for i in tqdm(range(num_eval)):
            framenum = choice(frameKeys)

            image = cv2.imread('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/moreframes/frame'+framenum+'.jpg')
            scores, locs = dataset.matchImage(image,None,0)
            if (locs[0][0]-frame_heading[framenum][0] <= 2) and locs[0][1]-frame_heading[framenum][1] <= 2:
                correctC += 1
            if locs[0][2] == frame_heading[framenum][2]:
                correctH += 1
            # print(correct)
        print('Cells: ' , correctC/float(num_eval))
        print('Headings: ' , correctH / float(num_eval))


if __name__ == '__main__':
    getAccuracy()
