# catkin_ws

###M and Rae's Notes on the code:


####Turtlebot Code (catkin_ws\src\speedy_nav\scripts)

__TurtleBot.py__ Four classes: one for the _TurtleBot_ object, and then three thread classes - _MovementControlThread_, _ImageSensorThread_, and _SensorThread_. (s2 in terms of units means seconds squared, we think -- in terms of seconds of an arc or whatever)
__TurtleBot__ just initialises all the threads and provides movement/utility methods - turning, bumper status, find angle to wall, and so on. Uses __Utils__. Uses CvBridge to be able to use images from kinect with OpenCV/ROS

__Utils__ All utils has in it is _getMedianAngle()_. Other utils are in __FixedActions__, which contains methods align, to move a fixed distance away from an object; _findAdjustedTargetArea_, which adjusts the expected size of the target given information about what angle it’s at; as well as _turnToNextTarget_ and _turnByAngle_, which are self-explanatory.

__CameraOnly__ is the same as __TurtleBot.py__ but without using the turtlebot so only uses Image and Depth sensors. Accesses a camera only (hence, the name). Has _ImageSensor_, _DepthSensor_, and _TurtleBot_ classes. Has _getMedianAngle()_. _TurtleBot_ initializes other 2 classes. Then, only handles finding angle to wall. 

__Planner__ has a _Planner_ class and an assisting _UpdateCamera_ class. The latter does pretty much what you would expect it to - refreshes images, saves them if the ‘t’ key is pressed, notices if the camera is stalled. 
Planner starts up the camera, requests a destination until it gets a valid one, tand then sets up a potential field brain with various behaviors from __FieldBehaviors__.

__FieldBehaviors__ includes a bunch of methods for use by a potential field brain: _RandomWander_, _KeepMoving_, _ObstacleForce_ (represents the repulsive force from an obstacle or barrier), and _ColorInvestigate_ (which is attracted to colors that the __MultiCamShift__ is interested in).

__MultiCamShift__ is interested in colors that are placed horizontally in a pattern (_getHorzPatterns_). Uses _getPotentialMatches_ to find the expected location of an interesting object given a couple instances of the same color horizontally but not vertically displaced.


#####Some useful tutorials:
TurtleBot - http://wiki.ros.org/Robots/TurtleBot

ROS explanation- http://robohub.org/ros-101-intro-to-the-robot-operating-system/

rospy - http://wiki.ros.org/rospy_tutorials

CvBridge - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython 



###Connecting to the TurtleBot using the workstation Enterprise:

First run though the necessary things to make sure the TurtleBot is started up correctly and the laptop is on. 

Ensure the laptop is connected to the `Macalester` network, not `macalester-guest`. The latter is outside the firewall and you won't be able to ssh in.

Open up a terminal on the laptop. Begin by getting the laptop's IP address with 
```
ifconfig
```
Then switch over to Enterprise and open a terminal. Run `ifconfig` again to get Enterprises's IP address and then run
```
python hostTurtlebotSetup.py
```
and enter both IPs when prompted. 

Next run 
```
source turtleros.bash
```
which should complete immediately with no output. Then you're ready to SSH into the laptop, by running
```
ssh -X macalester@<ip-address>
```
where `<ip-address>` is the address of the laptop. You will need to enter the password of the laptop, which is posted on the main board in the robot lab underneath the Robot Navigation poster. (The ssh should not take long, if it does check and make sure the laptop isn't conencted to `macalester-guest`. If it fails it'll time out after a minute or so.)

Then open a new terminal. In it, run 
```
roslaunch turtlebot_bringup minimal.launch
```
after which switch back to the first terminal and run 
```
roslaunch turtlebot_dashboard turtlebot_dashboard.launch &
```
You should now have the create dashboard up. 
The "&" symbol runs the dashboard in the background. You can then use the current terminal window to run the scripts you want to.


###Tips and Troubleshooting

If the makefile is not yet made properly go into the _catkin ws_ folder and then run 
```
catkin_make
```

If you're getting an error something like 
```
can't locate node in package ros
```
then you can try changing the permissions by running
```
chmod ugo+x Planner.py
```
or whatever file you need to change permissions on.
u, g, o are the three groups; r, w, x for read write and execute; + for adding permissions and - for taking them away.

To fix the issue with depth data, follow the advice from [this SO post](http://answers.ros.org/question/163551/no-depth-data-received-from-kinect-on-turtlebot/):

>There is a conflict with having both the depthclound and registered depthcloud launching at the same time. To view the depthcloud, set the registered depthcloud arguments to false when launching 3dsensors.launch

So run this instead of the usual bringup
```
roslaunch turtlebot_bringup 3dsensor.launch depth_registered_processing:=false depth_registration:=false
```
and it should work fine.

If you get an error something like "roslaunch: [ ] is neither a launch file in package [ ] nor is [ ] a launch file name" then you probably haven't sourced the setup file:
```
source devel/setup.bash
```
run when you're in the catkin directory.

If you're getting an error saying the process is crashing and you may not have the #! line in your file, it's actually because the #! line isn't where the program wants it - it does need to be the VERY FIRST line, even if there's just a block comment above it.

If you are on Enterprise trying to run programs on the TurtleBot try checking the host:
```
echo $ROS_MASTER_URI
```
If this returns some localhost path then you need to set the ros variables. Run
```
source turtleros.bash
```
Then you should be able to go back to happily running programs on the laptop from Enterprise. 

#####For a demo on how to work with ROS scripts: http://learn.turtlebot.com/

Very helpful short textbook intro to ROS: https://cse.sc.edu/~jokane/agitr/


