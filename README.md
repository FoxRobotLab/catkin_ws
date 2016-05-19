# catkin_ws
Works with TurtleBot

M and Rae's Notes on the code:


Turtlebot Code (catkin_ws\src\speedy_nav\scripts)

TurtleBot.py Four classes: one for the TurtleBot object, and then three thread classes - MovementControlThread, ImageSensorThread, and SensorThread. (s2 in terms of units means seconds squared, we think -- in terms of seconds of an arc or whatever)
TurtleBot just initialises all the threads and provides movement/utility methods - turning, bumper status, find angle to wall, and so on. Uses Utils. Uses CvBridge to be able to use images from kinect with OpenCV/ROS

Utils All utils has in it is getMedianAngle(). Other utils are in FixedActions, which contains methods align, to move a fixed distance away from an object; findAdjustedTargetArea, which adjusts the expected size of the target given information about what angle it’s at; as well as turnToNextTarget and turnByAngle, which are self-explanatory.

CameraOnly is the same as TurtleBot.py but without using the turtlebot so only uses Image and Depth sensors. Accesses a camera only (hence, the name). Has ImageSensor, DepthSensor, and TurtleBot classes. Has getMedianAngle(). TurtleBot initializes other 2 classes. Then, only handles finding angle to wall. 

Planner has a Planner class and an assisting UpdateCamera class. The latter does pretty much what you would expect it to - refreshes images, saves them if the ‘t’ key is pressed, notices if the camera is stalled. 
Planner starts up the camera, requests a destination until it gets a valid one, tand then sets up a potential field brain with various behaviors from FieldBehaviors.

FieldBehaviors includes a bunch of methods for use by a potential field brain: RandomWander, KeepMoving, ObstacleForce (represents the repulsive force from an obstacle or barrier), and ColorInvestigate (which is attracted to colors that the MultiCamShift is interested in).

MultiCamShift is interested in colors that are placed horizontally in a pattern (getHorzPatterns). Uses getPotentialMatches to find the expected location of an interesting object given a couple instances of the same color horizontally but not vertically displaced.


Some useful tutorials:
TurtleBot - http://wiki.ros.org/Robots/TurtleBot 
ROS explanation- http://robohub.org/ros-101-intro-to-the-robot-operating-system/
rospy - http://wiki.ros.org/rospy_tutorials
CvBridge - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython 


