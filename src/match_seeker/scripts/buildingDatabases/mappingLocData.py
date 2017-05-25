
import cv2



basePath = "/home/macalester/catkin_ws/src/match_seeker/"

numChange = open("workingNumChangeMapping.txt",'r')
data = open(basePath + "/res/locdata/Data-May25Thu-151653.txt",'r')
locations = open("locationsMay25-office2.txt",'w')
locData = {}

for pt in data.readlines():
    ptList = pt.split()
    # print ptList
    locData[ptList[0]] = ptList[1:]

for line in numChange:
    nums = line.split()
    locations.write(str(nums[0]) + " " + str(locData[nums[1]][0]) + " " + str(locData[nums[1]][1]) + " " + str(locData[nums[1]][2]) + "\n")


locations.close()
numChange.close()
data.close()



























