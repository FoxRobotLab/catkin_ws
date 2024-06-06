import datetime
import sys



heading_data = open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/06-18-2019-15-58-47newframes/compass-256.csv") ##CHANGE ME

frames = open("/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/06-18-2019-15-58-47newframes/06-18-2019-15-58-47frames.txt") ##CHANGE ME

frame_heading_file = open('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/markLocations/frameHeadingFile' + datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S') + '.txt', 'w')

def makeHeadingsRight(Heading):
    potentialHeadings = [0,45,90,135,180,225,270,315,360]
    currBest = 360
    best_h = 0
    dictionary = {45:315, 90:270, 135: 225, 225:135, 270:90, 315:45, 360:0}

    for h in potentialHeadings:
        difference = abs(Heading - h)
        if (difference < currBest):
            currBest = difference
            best_h = h

    if best_h in dictionary.keys():
        best_h = dictionary[best_h]

    return best_h


# retrieves milliseconds from compass data, formats to list of space separated elements
headingList = []
heading_lines = heading_data.readlines()
for line in heading_lines:
    newArray = line.split(",")
    millisec = newArray[0]
    cell = newArray[1]
    heading = newArray[2].strip()
    headingList.append(millisec + " " + cell + " " + str(makeHeadingsRight(int(float(heading)))))

frameList = frames.readlines()

# a list of arrays containing time/cell/heading pairings, in order named
time_cell_heading = []
for item in headingList:
    elem = item.split()
    time_cell_heading.append(elem)

# a list of arrays containing frame/time pairings, in order named
frame_time = []
for item in frameList:
    elem = item.split()
    elem[-1].strip()
    frame_time.append(elem)

for i in range(len(frame_time)):
    time = frame_time[i][1]
    lowest_diff = sys.maxint
    best_match = -1
    for j in range(len(time_cell_heading)):
        heading_time = time_cell_heading[j][0]
        diff = abs(int(time) - int(heading_time))
        if diff < lowest_diff:
            best_match = j
            lowest_diff = diff

    new_data = ""
    new_data += frame_time[i][0] + " "
    new_data += time_cell_heading[best_match][1] + " "
    new_data += time_cell_heading[best_match][2] + "\n"
    #show times
    #new_data += frame_time[i][1] + " "
    #new_data += time_cell_heading[best_match][0] + "\n"
    frame_heading_file.writelines(new_data)

heading_data.close()
frames.close()
frame_heading_file.close()
