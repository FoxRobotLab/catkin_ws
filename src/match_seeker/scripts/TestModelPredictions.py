""" -------------------------------------------------------------------------------------------------------------------
Tests the model predictions on a set of frames or a video from a data collection run, or a live run. It displays the images from the
robot alongside text stating the model's predictions on cell and heading. It also creates a file with the returned
predictions of the model and the actual cell and headings of each frame (currently, only for 'f' option)
Currently works for the 2024 LSTM models.
Works on Tensorflow and Keras 2.15.0 with Python 3.9, for reasons still unknown.
Text files are saved to src/match_seeker/res/classifier2022Data/DATA/Evaluation2024Data/Predictions/

Created: Summer 2024
Authors: Oscar Reza B. and Elisa Avalos
Edited: Summer 2026 by Jana Abu-Subha
------------------------------------------------------------------------------------------------------------------- """
import csv
import os
import time
from datetime import datetime

import cv2
from olri_classifier.cnnRunModel import ModelRunRGB
import OlinWorldMap
import socket
import struct
import numpy as np


class TestModelPredictions:
    def __init__(self):
        # Set data path for Precision 5820
        self.routename = None
        self.mainPath = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/"
        self.evalPath = os.path.join(self.mainPath, "Evaluation2024Data/")
        self.framesDataPath = os.path.join(self.evalPath, "FrameData/")

        # Set a directory for saving the predictions text file
        self.outputDir = os.path.join(self.evalPath, "Predictions/")

        # Set a directory for saving live accuracy data csv file
        self.csvPath = os.path.join(self.evalPath, "Accuracy/2026Data_RGBModel.csv") #change file according to model running

        # Load the model and the building map
        self.modelTester = ModelRunRGB()
        self.olinMap = OlinWorldMap.WorldMap()

        # Read the folder path
        self.framesFolderList = sorted(os.listdir(self.framesDataPath))
        print(f"{len(self.framesFolderList)} folders found!")

        self.folderName = ""
        self.folderPath = ""
        self.folderContents = []
        self.predictionFile = None
        self.imagesList = []
        self.linesList = []
        self.cell = None
        self.heading = None

        self.videoName = ""
        self.videoPath = ""
        self.videoCapture = None
        self.is_video = False

        self.robot = None
        self.is_live = False

        self.SERVER_IP = '141.140.243.153'  # set to robot's ip
        self.PORT = 5005

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.is_folder = False

    def userSelectFolder(self):
        """
        Creates an interactive terminal for user to select the frames folder to go through
        """
        filetype = input(f"Run test on .avi file, frames folder, or live robot run? (a/f/l): ")
        if filetype.lower() == "f":
            self.is_folder = True
            self.folderName, self.folderPath = self._selectDatasetFromList()
            self.folderContents = sorted(os.listdir(self.folderPath))
            self.getFramesAndAnnotations() # moved this call from main since currently this branch only applies to the folder of frames option
        if filetype.lower() == "a":
            self.is_video = True
            self.videoName, self.videoPath = self._selectDatasetFromList()
            self.folderName = self.videoName.replace(".avi", "")
            self.videoCapture = cv2.VideoCapture(self.videoPath)
        if filetype.lower() == "l":
            self.routename = input("Specify route (1/2/3/4/5 or 1B/2B/3B/4B/5B)")
            self.routename = self.routename.upper()
            self.is_live = True
            self.client_socket.connect((self.SERVER_IP, self.PORT))

    def _selectDatasetFromList(self):
        """
        Helper method to select a dataset (folder of frames/video file) from the list.
        """
        for folder in self.framesFolderList:
            if folder.endswith("frames") and not self.is_video:
                iterate = input(f"Frames folder: {folder}     Use this folder of frames? (y/n): ")
                if iterate.lower() == "y":
                    print('folder: ' + folder)
                    return folder, os.path.join(self.framesDataPath, folder)
                elif iterate.lower() == "n":
                    continue
                else:
                    print("Please input y/n")
                    exit(0)
            elif self.is_video and folder.endswith("avi"):
                iterate = input(f"Video: {folder}     Use this video? (y/n): ")
                if iterate.lower() == "y":
                    print('Video: ' + folder)
                    return folder, os.path.join(self.framesDataPath, folder)
                elif iterate.lower() == "n":
                    continue
                else:
                    print("Please input y/n")
                    exit(0)
        return "", ""

    def getFramesAndAnnotations(self):
        """
        Loads the video frames and their corresponding annotated text file
        """
        if self.is_folder:
            prediction_file_path = os.path.join(self.outputDir, self.folderName + "ModelPredictionsTest.txt")
            self.predictionFile = open(prediction_file_path, "w")
            # print(self.predictionFile)
            self.linesList = self._loadAnnotations()


    def _loadAnnotations(self):
        """Replace with server's actual IP if on another computer
        Helper method to load annotations from the corresponding text file.
        """
        annotFolder = os.path.join(self.evalPath, "AnnotData/")
        annotFolderList = sorted(os.listdir(annotFolder))
        for file in annotFolderList:
            #if file.endswith(self.folderName + ".txt"):
            if self.folderName in file and file.endswith(".txt"):
                return self._parseAnnotationFile(os.path.join(annotFolder, file))
        return []

    def _parseAnnotationFile(self, filePath):
        """
        Parses the annotation file and returns a list of lines.
        """
        lines = []
        with open(filePath) as textFile:
            for line in textFile:
                words = line.split(" ")
                words[5] = words[5].strip()
                lines.append(words[:6])
        return lines

    def displayPredictions(self):
        """
        Displays the predictions on the UI
        """
        frameCounter = 9
        read_index = 0
        missed_frames = 0
        while True:
            image, frame_name = self._getNextFrame(read_index)
            if image is None:
                missed_frames += 1
                print(f"Image not received. Retry {missed_frames}")
                if missed_frames >= 100:
                    print("100 consecutive frames not received, quitting program")
                    break
                time.sleep(0.1)
                continue
            missed_frames = 0
            self.imagesList.append(image)
            if len(self.imagesList) < 10:
                self._displayFrameWithoutPrediction(image)
                read_index += 1
                continue
            self._processAndDisplayFrame(image, frame_name, read_index)
            frameCounter += 1
            read_index += 1
        if self.is_folder:
            self.predictionFile.close()
        if self.is_video and self.videoCapture is not None:
            self.videoCapture.release()
        cv2.destroyAllWindows()
        print(f"Total processed frames: {frameCounter}")

    def _getNextFrame(self, index):
        """
        Helper method to displayPredictions which returns the next image to display
        """
        if self.is_live:
            try:
                # 1. Explicitly ask the server for the next frame
                self.client_socket.sendall(b"GET_FRAME")
                # 2. Read the 4-byte header to know the image payload size
                # header = self.client_socket.recv(5) # hard coded for now, will need to be dynamic
                header = self.client_socket.recv(16)
                if not header:
                    return None, None
                image_size = int(header.decode().strip())
                # 3. Receive the exact amount of image bytes (handles packet fragmentation)
                image_data = b""
                while len(image_data) < image_size:
                    packet = self.client_socket.recv(image_size - len(image_data))
                    if not packet:
                        break
                    image_data += packet
                # 4. Decode the JPEG binary data back into an OpenCV image array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame_name = f"live_frame_{index}.png"
                return image, frame_name
            except (UnicodeDecodeError, ValueError, socket.error) as e:
                print(f"Network or data parsing error: {e}")
                self._reconnectSocket()
                return None, None
        if self.is_video:
            for _ in range(60):
                success, temp_image = self.videoCapture.read()
                if not success:
                    return None, None
                image = temp_image
            frame_name = f"video_frame_{index}.png"
            return image, frame_name
        if index < len(self.folderContents):
            frame_name = self.folderContents[index]
            image = cv2.imread(os.path.join(self.folderPath, frame_name))
            return image, frame_name
        else:
            return None, None

    def _reconnectSocket(self):
        """Closes the broken socket and attempts to establish a clean connection."""
        print("Attempting to reconnect to robot server...")
        try:
            self.client_socket.close()
        except Exception:
            pass
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.settimeout(2.0)
            self.client_socket.connect((self.SERVER_IP, self.PORT))
            print("Reconnection successful!")
            return True
        except socket.error as e:
            print(f"Reconnection failed: {e}")
            return False

    def _displayFrameWithoutPrediction(self, image):
        """
        Displays the frame without prediction information.
        """
        # print(f"Cell prediction: {self.cell}")
        # print(f"Heading prediction: {self.heading}")
        cv2.imshow("frame", image)

    def _processAndDisplayFrame(self, image, frame, frameCounter):
        """
        Processes the frame to get predictions and displays the results.
        """
        scores, matchLocs = self.modelTester.getPrediction(self.imagesList, self.olinMap)
        if not matchLocs:
            print(f"Warning: No match locations returned for frame {frame}.")
            self.cell = "None"
            self.heading = "None"
        else:
            prediction = matchLocs[0]
            self.cell, self.heading = self._getPredictions(prediction)
        self._annotateImage(image)
        # For Folder of frames
        if self.is_folder or self.is_video:
            self._writePredictionsToFile(frame, frameCounter)
            cv2.imshow("frame", image)
            cv2.waitKey(60)
            self.imagesList.pop(0)
        if self.is_live:
            cv2.imshow("frame", image)
            record = cv2.waitKey(60)
            if record > -1 and chr(record).upper() == 'R':
                closeness = cv2.waitKey(0)
                new_row = []
                now = datetime.now()
                if chr(closeness).upper() == 'A':
                    new_row = [self.routename, now, self.cell, 0]
                elif chr(closeness).upper() == 'B':
                    new_row = [self.routename, now, self.cell, 1]
                elif chr(closeness).upper() == 'C':
                    new_row = [self.routename, now, self.cell, 2]
                elif chr(closeness).upper() == 'X':
                    new_row = [self.routename, now, self.cell, -1]
                else:
                    print("Invalid key")

                if new_row:
                    with open(self.csvPath, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(new_row)
            self.imagesList.pop(0)

    def _getPredictions(self, prediction):
        """
        Converts the prediction to cell and heading.
        """
        x_coord, y_coord, heading = prediction
        cell = self.olinMap.convertLocToCell((x_coord, y_coord))
        return cell, heading

    def _annotateImage(self, image):
        """
        Annotates the image with the cell and heading predictions.
        """
        image = cv2.putText(image, f"Cell prediction: {self.cell}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        # print(f"Cell prediction: {self.cell}")
        image = cv2.putText(image, f"Heading prediction: {self.heading}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        # print(f"Heading prediction: {self.heading}")

    def _writePredictionsToFile(self, frame, frameCounter):
        """
        Writes the predictions to the prediction file.
        """
        if self.predictionFile: #.avi files and live turtle footage wont have a prediction file
            self.predictionFile.write(
                f"{frame}  Predictions -- Cell: {self.cell}   Heading: {self.heading}\n"
                f"        Actual -- Cell: {self.linesList[frameCounter][3]}   Heading: {self.linesList[frameCounter][4]}\n"
            )

    # def _createPredictionDictionary(self):
    #     """
    #     TODO: Implement this +method to put all the information in a pandas dataframe for easy access
    #     """
    #     data_dictionary = pd.DataFrame(
    #         {
    #             "Frame": "",
    #             "Cell Prediction": 0,
    #             "Cell Actual": 0,
    #             "Cell Correct": False,
    #             "Heading Prediction": 0,
    #             "Heading Actual": 0,
    #             "Heading Correct": False,
    #             "All Correct": False,
    #         }
    #     )

if __name__ == "__main__":
    testPredictor = TestModelPredictions()
    testPredictor.userSelectFolder()
    testPredictor.displayPredictions()
