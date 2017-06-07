
origLocData = open("/home/macalester/catkin_ws/src/match_seeker/res/locations/052517LOCATIONS.txt", "r")
deleteData = open("/home/macalester/catkin_ws/src/match_seeker/scripts/buildingDatabases/toDelete.txt", "r")
newLocData = open("/home/macalester/catkin_ws/src/match_seeker/res/locations/create0607.txt", "w")

deleteList = []

for line in deleteData.readlines():
    line = line.strip()
    deleteList.append(str(line))


for line in origLocData.readlines():
    data = line.split()
    if str(data[0]) not in deleteList:
        newLocData.write(line)



origLocData.close()
deleteData.close()
newLocData.close()

