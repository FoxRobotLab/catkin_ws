""" ========================================================================
DataPaths.py

This defines the base path where this code is, so we can switch between
computers more easily.

Update 2019: only correct/necessary for map paths

======================================================================== """




import os

# base path on Susan's computer
basePath = "/Users/susan/Desktop/ResearchStuff/Summer2016-2017/GithubRepositories/catkin_ws/src/match_seeker/"
graphMapData = "res/map/cellGraph.txt"
mapLineData = "res/map/olinNewMap.txt"
cellMapData = "res/map/mapToCells.txt"


# base path on Enterprise
# basePath = "/home/macalester/catkin_ws/src/match_seeker/"


# base path on FoxVoyager (aka Speedy)
# basePath = "/home/macalester/Desktop/githubRepositories/catkin_ws/src/match_seeker/"

# base path on Precision
# basePath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/"
# #
# turtleBase = 'kobuki' # os.environ["TURTLEBOT_BASE"]
# #
# if turtleBase == "create":
#     imageDirectory = "res/create060717/"
#     locData = "res/locations/SUSANcreate0607.txt"
# elif turtleBase == "kobuki":
#     # imageDirectory = "res/kobuki0615/"
#     imageDirectory = "res/allFrames060418/"
#     # locData = "res/locations/Susankobuki0615.txt"
#     locData = "res/locdata/allLocs060418.txt"
#
# # graphMapData = "res/map/olinGraph.txt"
# graphMapData = "res/map/cellGraph.txt"
# mapLineData = "res/map/olinNewMap.txt"
# cellMapData = "res/map/mapToCells.txt"
