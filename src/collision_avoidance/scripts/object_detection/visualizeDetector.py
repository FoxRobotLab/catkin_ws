"""--------------------------------------------------------------------------------
visualizeDetector.py
Author: Oscar Reza B.

Reads the frames as encoded text from sendFrames.py to use as input for YOLO26
model. It displays them using cv2.

To run this file:
1. Cutie2 must be launched (minimal, astra, and teleop)
2. In terminal run command:
      rosrun collision_avoidance scripts/object_detection/sendFrames.py
3. Run this script through Pycharm IDE.

To end this program (if it does not end as expected),
processes must be killed in the terminal using kill -9 <Python2.7 PID>

--------------------------------------------------------------------------------"""
import base64
import os
import time

import cv2
import zmq
import numpy as np

from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")  # lightweight model

# Setup ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, '')


def saveFrameToFolder(img, folderName, currFrameNum):
    fName = "frame{}.jpg".format(time.strftime("%Y%m%d-%H%M%S"))
    pathAndName = folderName + fName
    try:
        cv2.imwrite(pathAndName, img)
    except:
        print("Error writing file", currFrameNum, pathAndName)


def saveLabelToFolder(folderName, currFrameNum, detectedPeople):
    fName = "frame{}.txt".format(time.strftime("%Y%m%d-%H%M%S"))
    pathAndName = folderName + fName
    try:
        with open(pathAndName, "w") as f:
            for detectedPerson in detectedPeople:
                f.write(f"{detectedPerson}\n")
    except:
        print("Error writing file", currFrameNum, pathAndName)


if __name__ == "__main__":
    # Define destination directory
    destDir = "/home/macalester/PycharmProjects/catkin_ws/src/collision_avoidance/res/mock_ds"

    # Create directory for storing frames
    timestamp = "{}".format(time.strftime("%Y%m%d-%H%M"))
    frameFolder = destDir + '/images/' + timestamp + 'frames/'
    os.mkdir(frameFolder)

    # Create directory for storing labels
    timestamp = "{}".format(time.strftime("%Y%m%d-%H%M"))
    labelsFolder = destDir + '/labels/' + timestamp + 'frames/'
    os.mkdir(labelsFolder)

    # Initialize counters
    frameNum = 0
    frameSaveCount = 0

    while True:
        # Receive message from socket
        msg = socket.recv()

        # Decode message into image
        jpg_original = base64.b64decode(msg)
        np_arr = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO and display results
        results = model.predict(source=frame, classes=[0], conf=0.75, verbose=False)[0]
        annotated = results.plot()
        cv2.imshow("YOLO Detection", annotated)

        # Get bounding boxes
        boxes = results.boxes.xyxy.tolist()

        detectedPedestrians = []
        # Print info
        if len(boxes) > 0:
            print("Person Id |  x0   |  y0   |  x1   |  y1   ")
            for i, box in enumerate(boxes, start=1):
                x0 = box[0]
                y0 = box[1]
                x1 = box[2]
                y1 = box[3]
                print(f"    {i}     | {x0:3.2f} | {y0:3.2f} | {x1:3.2f} | {y1:3.2f} |")

                predLabel = f"{i} movementStatus {x0:3.2f} {y0:3.2f} {x1:3.2f} {y1:3.2f} theta 1.75 0.83 0.25 longDistInM latDistInM"
                detectedPedestrians.append(predLabel)

        # Save frame without prediction boxes and save annotations separately
        frameSaveCount += 1
        if frameSaveCount == 10:
            saveFrameToFolder(frame, frameFolder, frameNum)
            saveLabelToFolder(labelsFolder, frameNum, detectedPedestrians)
            frameNum += 1
            frameSaveCount = 0

        # Handle quitting
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
