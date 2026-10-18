# Spring 2026 Semester Project
### Oscar Reza Bautista and Ryan Maule

## About this directory
This folder contains the resources relevant to this semester project.

### Structure of this project

```
| [collision_avoidance]
|-- scripts/
    |-- data_collection/
        |-- annotateData.py
        |-- calculateLatLong.py
        |-- collectData.py
        |-- dataGenerator.py
        |-- fixAnnotations.py
    |-- object_detection/
        |-- data_collection
        |-- data_collection
        |-- data_collection
        |-- data_collection
        |-- data_collection
        |-- data_collection
    |-- training/
        |-- data_collection
```

### Data Collection
#### collectData.py
- A testing script, not used to build the final datasets
- Written in Python 2.7
- Runs on Turtlebot2, only tested in Cutie2
- Reads images from robot's camera to create the frames and labels directories

#### calculateLatLong.py
- Takes in the directory for the frame labels
- Modifies each line with predict longitudinal and latitudinal distances 
from the robot to the detected pedestrians in meters

#### fixAnnotations.py
- Takes in frames and labels paths
- Opens interactive GUI to manually annotate frames
- Allows to add undetected pedestrians
- Here's where we add direction and movement status, as well as known person
measurements

#### dataGenerator.py
- Assumes a specific structure
- Prepares training and validation splits for different YOLO detection tasks

#### annotateData.py
- Does not save any new files
- Takes in the paths for frames and labels
- Opens a cv2 window with drawn bounding boxes for the detected 
pedestrians


### Object Detection
#### calibration.py
- Generates the homography matrix and exports the `homography.npy`

#### classifyRisk.py
- Computes the collision risk for the detected pedestrians and visualizes them
using OpenCV
- Requires having an homography path, computed in `calibration.py`
- Takes in a `save_flag` to save the annotated frames

#### sendFrames.py
- Reads the robot's live camera feed
- Runs on the bot through SSH

#### visualizeDetector.py
- This is the actual data collection script
- Runs on the Precision Tower or personal laptop
- Requires `sendFrames.py` running on the bot
- Runs YOLO object detection on the live images from the robot
- Displays detected pedestrians and writes the coordinates as labels
- Saves the frames and the labels in their respective directories

### Model Training
The file `trainDetectors.py` has some code to fine-tune YOLO26 for our tasks of
interest. However, the actual training for this project was done on a Google
Colab notebook.
- [Link to the Colab Notebook for model training](https://colab.research.google.com/drive/1dJhZ-mNJHepy-rpXi82hOAav6vSK9n9k)
