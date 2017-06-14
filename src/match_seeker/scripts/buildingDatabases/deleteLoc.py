
origLocData = open("/home/macalester/catkin_ws/src/match_seeker/res/locations/locationsMay30.txt", "r")
deleteData = open("/home/macalester/catkin_ws/src/match_seeker/scripts/buildingDatabases/toDeleteKobuki.txt", "r")
newLocData = open("/home/macalester/catkin_ws/src/match_seeker/res/locations/kobuki0609.txtOLD", "w")

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

