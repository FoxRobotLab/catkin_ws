with open('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/map/mapToCells.txt') as mapToCells:
    lines = mapToCells.readlines()
    for line in lines:
        if line[0] is not '#':
            l = line.split()
            print("{0} ({1:.1f}, {2:.1f})".format(l[0],(float(l[1])+float(l[3]))/2,(float(l[2])+float(l[4]))/2))
