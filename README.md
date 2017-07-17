# catkin_ws

M, Rae, Kayla, and Xinyu's Notes on the code:


## MEET YOUR ROBOT PALS:
Speedy is a Turtlebot mk 1. They are off-white with an upgraded Dell laptop strapped on top. Their odometer cannot be reset between start-ups, and their charging processes are finicky. Speedy was the first robot associated with these programs, so some of the older versions will only run with them. Later versions have been modified to run on either Turtlebot.

Cutie is a Turtlebot mk 2. They are sleeker and lighter than Speedy, but sometimes they react to lower level code in surprising ways since it was fine tuned on Speedy. Cutie is shiny and black with a netbook on top, and they have a nifty charging dock that’s easy to connect. 

## Turtlebot Code 

All code for this project is located under catkin_ws/src. In this directory, there are three important folders: 
**_speedy_nav_**, **_qr_seeker_**, and **_match_seeker_**. All of these contain programs intended to make the robot navigate independently around Olin-Rice. In **_speedy_nav_**, this is done by a bunch of tri-colored signs (i.e. green-blue-violet), where the color pattern tells the robot where it is. Due to the difficulty of color tracking, the robot cannot read these signs with any degree of reliability. Additionally, as it navigates the corridors it must continually swivel and look around for the signs, which is slow. **_qr_seeker_** aimed to solve these problems by using only some color recognition - blue signs, the color most reliably recognised - along with QR codes, which is arguably effective. Building upon **_qr_seeker_**, **_match_seeker_** uses the combination of image data with location tags, odometry information and monte carlo localization to better instruct the robot to navigate.

## Booting up
Before you launch **_match_seeker_**, be sure to launch all the necessary nodes on the laptop and the base station. This includes running three launches on the laptop:
> roslaunch turtlebot_bringup minimal.launch

On Cutie:
> roslaunch turltebot_bringup 3dsensor.launch

Or on Speedy:
> roslaunch turtlebot_bringup 3dsensor.launch depth_registered_processing:=false depth_registration:=false

And the speaking node:
>rosrun match_seeker speaker.py

On the Enterprise base station, you should launch the dashboard and the message node (the talking correspondent to the speaking node on the laptop):
roslaunch turtlebot_dashboard turtlebot_dashboard.launch
rosrun match_seeker message.py


## Introducing main programs
**_match_seeker_**(catkin_ws/src/match_seeker/scripts)

Note: Unlike **_qr_seeker_**, match_seeker does not require webcams attached to the platform on both sides of the robot.
**matchPlanner.py** is the heart of the program. It communicates with **Localizer.py** to get more information on the robot’s location. It also communicates with the robot and give the robot instructions.
**Localizer.py** helps the robot to locate itself with either image matching or monte carlo localizer in charge. It has the data that **matchPlanner.py** doesn’t, which includes data from **imageDataset.py** and **MonteCarloLocalize.py**. Whether the robot is navigated by images or MCL is decided by the variance of Monte Carlo localization. If the variance is greater than 5, we would assume that the possible locations were too spread out and MCL is not confident enough about where it is, so the robot will be navigated by images. Otherwise, we believe that MCL shows a stronger reliability and the robot should be navigated by the MCL.

For the image matching part of the program, **ImageDataset.py** is the main program, and there is also helper program such as **ImageFeatures.py**. 
**ImageDataset.py** creates the ImageDataset object, which reads in a set of images and a location file, and creates dictionaries and a KDTree to hold the images and organize them by their locations. It also provides methods for getting some number of closest matches.
**ImageFeatures.py** collects up different features of the image and can display the results, or compare to another for similarity.

**MonteCarloLocalize.py** is used to implement Monte Carlo localization. It generates a lot of random possible locations of the robot at first. Each particle indicates one possible location and is a class object generated and organized by **Particle.py**. Then **MonteCarloLocalize.py** moves those particles according to the movement of the robot, which is gathered from **odometryListener.py**, and eliminates the particles that are no longer possible (such as those that hit a wall). It gives weights to all the particles according to their possibility, and calculates a center of mass particle by the normalized weights, which should be the most possible location of the robot. When the MCL information is off, there is a scatter function that can be used in **MonteCarloLocalize.py**, which, for every particle, will generate a new particles in a uniform area around the center of mass with a random heading.

**SeekerGUI.py** is a graphical user interface to display the information in a more comfortable way compared with the terminal. It has the ability to generate a popup window that asks for a input destination node number, then this piece of information will be passed onto **matchPlanner.py** for further instructions. It displays a window with five main components.
    The green part includes information about navigation mode (either navigating or localizing) and whether it’s navigated by images or MCL.
    The blue part exhibits x, y, and heading for odometry information, last known location, MCL center of mass, and the three best matched images. It also contains information about confidence.
    The yellow part prints out messages that were in the terminal.
    The purple frame has the information about image matching, which includes closest node , next node, also target distance and search radius. It also prints out match status.
    The pink part includes information on the turning, which is made up of turn status, current heading, target angle, and turn angle.



## Database Management:
The first step in picking up this project again will be to update the image dataset. Although **_match_seeker_** uses a combination of many types of data to navigate, it is still heavily dependent on camera input. There are a number of programs under _match_seeker/scripts/buildingDatabases_ and _match_seeker/scripts/markLocations_ for keeping the dataset up to date. These programs should be run in python. We found it easiest to import the images into labelled folders under match_seeker/res, since they need to be accessed by the program, but the database programs do work on paths that lead outside of catkin_ws. 

First, gather data by running the robot around Olin-Rice under teleop (http://wiki.ros.org/turtlebot_teleop/graveyard/Tutorials/Teleoperation)  from the Enterprise base station and save the images. This can either be done by taking video or saving individual frames at important locations. There are more instructions on video here:
http://learn.turtlebot.com/2015/02/04/4/

If you just want to take stills, stop after you open the image_view and follow instructions there for saving stills. 
If you save images as a video, you will need to run **classVideoToFiles.py** to convert the frames to jpgs.
**scanImageMatches.py** uses the same image matching program as the program to find matching images in the dataset. If you took video, there will be dozens of frames saved at every location, and you just don’t need to have that many images in the database. This program displays two images. If they match, press “y” to record the number of the matching image in a txt file. Press any other key to advance. NOTE: this program assumes that the first image in a series of matching images is the best one, but that is often not the case since the camera image gets very blurry during turning. We found it easiest to have pen & paper nearby either while doing this or just after and recording blurry image numbers while passing over them during deletion. You can then add your list of blurry images to the list of matches compiled by this program and delete them all at once.

**createDatabase.py** creates a copy of an image database that does not include the images named in a txt file. We used this to removed the duplicate images from our database. Make sure to update the paths at the top of this file.

**organizePics.py** renames files in consecutive order and moves them from one folder to another. It also writes a txt file cataloging the old name of every image and its new name. This is very useful if you have deleted a lot of files throughout the database or you need to combine two smaller groups of images into one. Make sure to update the paths at the top and the startNum on line 61. startNum is equal to 0 if you’re renumbering a standalone directory, or to the highest image number in the folder you want to move your pictures into (which will become the name of the first file in your current directory, since images index at 0). Images do not have to be named in exact consecutive order, but it is important that there are no duplicate names to avoid overwriting the database. NOTE: If you are not combining image folders, you need to set your images to move into an empty folder. Be careful when moving images into a folder that already has images in it, because this program does not keep a copy of the original folder and it is not easily undone. If you mess up you may need to rerun image deletion and renaming on multiple old versions of your images.

**mappingLocData.py** uses the name change file written by organize pics to update location data txt files for any images that have been renamed.

**doubleNameChange.py** is an emergency program for just in case you ran organizePics twice before mappingLocData. It writes a name change txt file that maps an images original name to its third name.

## markLocations
**locsForFrams.py** is the first step of location tagging. It can be run on video (if you want to tag locations before removing duplicate images) or on a folder of jpgs. This program displays a map, an image, and a grid of possible angular headings. For each image you can adjust the robot’s location on the map with the WASD keys, adjust its heading by clicking on the grid, and advance with the spacebar. The robot cameras have fish eye lenses, so the always appear to be a few feet ahead of where they actually are. Typically, if the base of a wall is along the bottom of the image, the robot is in the center of the hallway. If a doorframe you are driving past is close to the edge of the image, the robot is in the middle of that doorway.

**checkLocs.py** is for double checking your existing location tags. YOU SHOULD RUN CHECKLOCS EVERY TIME YOU TAG LOCATIONS. It has almost the same controls as locsForFrames, but uses X/C to change angles. Also, checkLocs keeps up with an ongoing offset, so sometimes after you move one node forward a few feet, you may need to move the next one back. 

**checkNode.py** lets you check location tags around a specific x,y coordinate. 

This folder has a readMap file because pycharm is finicky about importing classes from other folders, and there are a few more programs in here about classifying types of images (ie hallway, atrium, etc) that might be part of future applications for this project.

## previous files
**_qr_seeker_** (catkin_ws/src/qr_seeker/scripts)

IMPORTANT: **_qr_seeker_** assumes that the robot has been augmented with webcams, taped/somehow attached to the platform the kinect is mounted on. It won't work if you don't have these webcams! It's also possible that the numbers opencv assigns to them will change - usually the laptop's webcam is 0 and then the USB webcams are 1 and 2, but if you're getting errors that look like the camera isn't found, try messing with the numbers in the sample program **multipleCameras.py** (catkin_ws/src/qr_seeker/scripts/testing scripts/multipleCameras.py). If you're getting the error that the device has no space left try changing the FPS and Frame size for the webcams. (Also, if you happen to be using the Dell machine, be aware that the fourth 'USB port' is actually the eSATA port on the right of the machine - we plugged in the kinect and one of the webcams into the two USBs on the right, then the other webcam into the USB on the left, and the Create into the eSATA on the right. This let us have better resolution on both webcams.)

Also important: the master branch is what you want to work with. The orb branch is a slightly clunky offshoot with tools for automatically taking images and comparing to lots of references, the idea being to try using only ORB with a bunch of reference images.

**qrPlanner.py** is the heart of the program. It directs sign/QR code seeking and aligning and controls the potential field brain. It also controls all three cameras.

**ORBrecognizer.py** and **QRrecognizer.py** are classes which contain a scanner for ORB features or QR codes, respectively, and have a bunch of methods those scanners need. ORBrecognizer.py has a bunch of image pre-processing. Since it's only looking for these blue signs, it masks out everything in the image that is not blue, and only looks for keypoints in the remaining image. This means we're not looking for keypoints on every chair or garbage bin we pass, which improves accuracy. Once it finds keypoints, it determines which of them are good and draws a picture of the matches between the image and the reference. It returns the good keypoints so that other functions can do things like find the moment of them. **QRrecognizer.py** does a lot less: pretty much all it does it look for a QR code, and then if it finds an appropriate one, formats the output (from the code's string) prettily and hands it back. The code is pretty gross just because zbar is gross - you don't really need to understand it, but be aware that the weird useless-looking for loop with only a pass inside of it is necessary! For some bizarre reason, zbar stores the result of the scan in the last symbol in image, so the for loop is just to get symbol set to the result.

_scripts/testing scripts_

The code in here should be run using python, not roslaunched. It's just bits and pieces that are helpful if you need to do things like test camera settings or figure out which numbers your cameras have.

A note about the cameras: Apparently, Linux is not happy if you have multiple webcams that aren't on different USB buses, because they take up all the - bandwidth? So as a solution, we've manually dropped the framerate and resolution. You can play with these values in **multipleCameras.py** and **qrMultCam.py**. We weren't able to get the framerate above 12 without dropping the resolution. A framerate of 30 and size of 480x360 works, but we got much better results with the higher resolution and lower framerates -- dropping the resolution means you need to have bigger QR codes and look for fewer ORB points, and the robot can't see the codes from across the hall. If the cameras suddenly stop working at settings that worked before, or if you can't get full res images on the same hardware with a framerate of 12 frames/sec, try restarting the laptop, unplugging and plugging everything back in. This has always worked for us.

**_speedy_nav_** (catkin_ws/src/speedy_nav/scripts)

**TurtleBot.py** 

Four classes: one for the TurtleBot object, and then three thread classes - MovementControlThread, ImageSensorThread, and SensorThread. (s2 in terms of units means seconds squared, we think -- in terms of seconds of an arc or whatever) TurtleBot just initialises all the threads and provides movement/utility methods - turning, bumper status, find angle to wall, and so on. Uses Utils. Uses CvBridge to be able to use images from kinect with OpenCV/ROS

**Utils**

All utils has in it is _getMedianAngle()_. Other utils are in *_FixedActions_, which contains methods align, to move a fixed distance away from an object; _findAdjustedTargetArea_, which adjusts the expected size of the target given information about what angle it’s at; as well as turnToNextTarget and turnByAngle, which are self-explanatory.

CameraOnly is the same as **TurtleBot.py** but without using the turtlebot so only uses Image and Depth sensors. Accesses a camera only (hence, the name). Has ImageSensor, DepthSensor, and TurtleBot classes. Has **getMedianAngle()**. TurtleBot initializes other 2 classes. Then, only handles finding angle to wall.

Planner has a Planner class and an assisting UpdateCamera class. The latter does pretty much what you would expect it to - refreshes images, saves them if the ‘t’ key is pressed, notices if the camera is stalled. Planner starts up the camera, requests a destination until it gets a valid one, tand then sets up a potential field brain with various behaviors from FieldBehaviors.

FieldBehaviors includes a bunch of methods for use by a potential field brain: RandomWander, KeepMoving, ObstacleForce (represents the repulsive force from an obstacle or barrier), and ColorInvestigate (which is attracted to colors that the MultiCamShift is interested in).

MultiCamShift is interested in colors that are placed horizontally in a pattern (getHorzPatterns). Uses getPotentialMatches to find the expected location of an interesting object given a couple instances of the same color horizontally but not vertically displaced.

## Connecting to the TurtleBot using the workstation Enterprise:
First run through the necessary things to make sure the TurtleBot is started up correctly and the laptop is on.
Ensure the laptop is connected to eduroam network. It should have a fixed IP from ITS but it doesn’t hurt to check now and then with
ifconfig


Then switch over to Enterprise and open a terminal. Run ifconfig again to get Enterprises's IP address and then run
python hostTurtlebotSetup.py


and enter the number associated with which Turtlebot you’re using and both IPs when prompted.
Next run
source .bashrc in all used terminals on Enterprise.


which should complete immediately with no output. Then you're ready to SSH into the laptop, by running
ssh -X macalester@<ip-address>


where <ip-address> is the address of the laptop. You will need to enter the password of the laptop, which is posted on the main board in the robot lab underneath the Robot Navigation poster. (The ssh should not take long, if it does check and make sure the laptop is on the right network. If it fails it'll time out after a minute or so.)
Then open a new terminal. In it, run
roslaunch turtlebot_bringup minimal.launch


after which switch back to the first terminal and run
roslaunch turtlebot_dashboard turtlebot_dashboard.launch &


You should now have the create dashboard up. The "&" symbol runs the dashboard in the background. You can then use the current terminal window to run the scripts you want to.


## Tips and Troubleshooting

Makefiles If the makefile is not yet made properly go into the catkin ws folder and then run
> catkin_make

If you're getting an error something like can't locate node in package ros, then you can try changing the permissions by running
chmod ugo+x Planner.py or whatever file you need to change permissions on. u, g, o are the three groups; r, w, x for read write and execute; + for adding permissions and - for taking them away.

_Depth Data _
To fix the issue with depth data on Turtlebot 1 (Speedy), follow the advice from this SO post:

There is a conflict with having both the depthclound and registered depthcloud launching at the same time. To view the depthcloud, set the registered depthcloud arguments to false when launching 3dsensors.launch
So run this instead of the usual bringup

> roslaunch turtlebot_bringup 3dsensor.launch depth_registered_processing:=false depth_registration:=false

and it should work fine.

Source If you get an error something like "roslaunch: [ ] is neither a launch file in package [ ] nor is [ ] a launch file name" then you probably haven't sourced the setup file:

> source.bashrc

run when you're in the catkin directory.

Need #! in Line If you're getting an error saying the process is crashing and you may not have the #! line in your file, it's actually because the #! line isn't where the program wants it - it does need to be the VERY FIRST line, even if there's just a block comment above it.

unning Programs from Enterprise If you are on Enterprise trying to run programs on the TurtleBot try checking the host:

> echo $ROS_MASTER_URI

If this returns some localhost path then you need to set the ros variables. Run

> source .bashrc

Then you should be able to go back to happily running programs on the laptop from Enterprise.

Check ROS Variables on Workstation and Laptop To check the environment variables on the workstation, run

> echo $ROS_IP

You should get the IP address of the workstation back.
Ensure that on the laptop the ROS HOSTNAME and the ROS IP are set to the laptop. To do this run the above command and:

> echo $ROS_HOSTNAME

If they are not set, run:
> source .bashrc

 
## Some useful tutorials and links:
TurtleBot - http://wiki.ros.org/Robots/TurtleBot
ROS explanation- http://robohub.org/ros-101-intro-to-the-robot-operating-system/
rospy - http://wiki.ros.org/rospy_tutorials
CvBridge - http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
The laptop should then set the necessary variables. You can also look at tips at http://wiki.ros.org/Robots/TurtleBot/Network%20Setup.
For a demo on how to work with ROS scripts: http://learn.turtlebot.com/
Very helpful short textbook intro to ROS: https://cse.sc.edu/~jokane/agitr/
For github readme font information: https://help.github.com/articles/basic-writing-and-formatting-syntax/
