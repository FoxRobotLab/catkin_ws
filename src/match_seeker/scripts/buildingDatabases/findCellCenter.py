#
# with open('/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/map/mapToCells.txt') as cells:
#     lines = cells.readlines()
#
#     for line in lines:
#         arr = line.split()
#         if arr[0] != '#':
#             centerX = ( float(arr[1]) + float(arr[3]) )/ 2
#             centerY = (float(arr[2]) + float(arr[4])) / 2
#             print(arr[0] + ' ('+str(centerX)+', ' + str(centerY)+')')


for i in range(200,219):
    print(str(i) +' '+ str(i+1))
