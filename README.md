# catkin_ws

###M and Rae's Notes on the code:


####Turtlebot Code 

All code is located under `catkin_ws/src`. In this directory, there are two important folders: `speedy_nav` and `qr_seeker`. Both of these contain programs intended to make the robot navigate independently around Olin-Rice. In `speedy_nav`, this is done by a bunch of tri-colored signs (i.e. green-blue-violet), where the color pattern tells the robot where it is. Due to the difficulty of color tracking, the robot cannot read these signs with any degree of reliability. Additionally, as it navigates the corridors it must continually swivel and look around for the signs, which is slow. `qr_seeker` aims to solve these problems by using only some color recognition - blue signs, the color most reliably recognised - along with QR codes. This is arguably effective.


#### `qr_seeker` (`catkin_ws/src/qr_seeker/scripts`)
IMPORTANT: `qr_seeker` assumes that the robot has been augmented with webcams, taped/somehow attached to the platform the kinect is mounted on. It won't work if you don't have these webcams! It's also possible that the numbers opencv assigns to them will change - usually the laptop's webcam is 0 and then the USB webcams are 1 and 2, but if you're getting errors that look like the camera isn't found, try messing with the numbers in the sample program `multipleCameras.py` (`catkin_ws/src/qr_seeker/scripts/testing scripts/multipleCameras.py`). (Also, if you happen to be using the Dell machine, be aware that the fourth 'USB port' is actually the eSATA port on the right of the machine - we plugged in the create and kinect into the two USBs on the right, and then the webcams into the USB on the left and the eSATA on the right.)


__qrPlanner.py__ is the heart of the program. It directs sign/QR code seeking and aligning and controls the potential field brain. It also controls all three cameras.


#### `scripts/testing scripts`
The code in here should be run using python, not roslaunched. It's just bits and pieces that are helpful if you need to do things like test camera settings or which numbers your cameras have. 

A note about the cameras: Apparently, Linux is not happy if you have multiple webcams that aren't on different USB buses, because they take up all the - bandwidth? So as a solution, we've manually dropped the framerate and resolution. You can play with these values in `multipleCameras.py` and `qrMultCam.py`. We weren't able to get the framerate above 12 without dropping the resolution, but the a framerate of 30 and size of 480x360 worked well for us. Note that dropping the resolutoin will mean you may need to have bigger QR codes and look for fewer ORB points.


#### `speedy_nav` (`catkin_ws/src/speedy_nav/scripts`)

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

####Makefiles
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

####Depth Data
To fix the issue with depth data, follow the advice from [this SO post](http://answers.ros.org/question/163551/no-depth-data-received-from-kinect-on-turtlebot/):

>There is a conflict with having both the depthclound and registered depthcloud launching at the same time. To view the depthcloud, set the registered depthcloud arguments to false when launching 3dsensors.launch

So run this instead of the usual bringup
```
roslaunch turtlebot_bringup 3dsensor.launch depth_registered_processing:=false depth_registration:=false
```
and it should work fine.

####Source
If you get an error something like "roslaunch: [ ] is neither a launch file in package [ ] nor is [ ] a launch file name" then you probably haven't sourced the setup file:
```
source devel/setup.bash
```
run when you're in the catkin directory.

####Need #! in Line
If you're getting an error saying the process is crashing and you may not have the #! line in your file, it's actually because the #! line isn't where the program wants it - it does need to be the VERY FIRST line, even if there's just a block comment above it.

####Running Programs from Enterprise 
If you are on Enterprise trying to run programs on the TurtleBot try checking the host:
```
echo $ROS_MASTER_URI
```
If this returns some localhost path then you need to set the ros variables. Run
```
source turtleros.bash
```
Then you should be able to go back to happily running programs on the laptop from Enterprise. 

####Check ROS Variables on Workstation and Laptop
To check the envirionment variables on the workstation, run 
```
echo $ROS_IP
```
You should get the IP address of the workstation back. 

Ensure that on the laptop the ROS HOSTNAME and the ROS IP are set to the laptop. To do this run the above command and:
```
echo $ROS_HOSTNAME
```
If they are not set, run:
```
source .bashrc
```
The laptop should then set the necessary variables.  You can also look at tips at http://wiki.ros.org/Robots/TurtleBot/Network%20Setup. 

#####For a demo on how to work with ROS scripts: http://learn.turtlebot.com/

Very helpful short textbook intro to ROS: https://cse.sc.edu/~jokane/agitr/


