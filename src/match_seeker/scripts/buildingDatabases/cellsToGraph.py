
basePath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/"

filename = basePath + "res/map/mapToCells.txt"


def processLines(fObj):
    for line in fObj:
        if line == "" or line.isspace() or line[0] == '#':
            continue
        else:
            parts = line.split()
            cellNum = int(parts[0])
            [x1, y1, x2, y2] = [float(v) for v in parts[1:]]
            avgX = (x1 + x2) / 2
            avgY = (y1 + y2) / 2
            print cellNum, (avgX, avgY)



fil = open(filename)
processLines(fil)

