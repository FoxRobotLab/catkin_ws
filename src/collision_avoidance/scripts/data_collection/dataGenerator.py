"""
File: dataGenerator.py
Authors: Oscar Reza B. and Ryan Maule
Created: Spring 2026

This script prepares the training and validation splits for different YOLO detection tasks.

Required file structure:
res/
 |-- dataset_name/
     |-- images/
         |-- 2026XXXX-XXXXframes/
             |-- frame2026XXXX-XXXXXX.jpg
     |-- labels/
         |-- 2026XXXX-XXXXframes/
             |-- frame2026XXXX-XXXXXX.txt

The output files structure looks like this:
res/
 |-- dataset_name_task_name/
     |-- data.yaml
     |-- train/
         |-- images/
             |-- frame2026XXXX-XXXXXX.jpg
         |-- labels/
             |-- frame2026XXXX-XXXXXX.txt
     |-- valid/
         |-- images/
             |-- frame2026XXXX-XXXXXX.jpg
         |-- labels/
             |-- frame2026XXXX-XXXXXX.txt
"""

import os
import shutil
import random

# Define task name ("status", "direction", or "all"), modify as needed
TASK_NAME = "direction"

# Define paths, modify as needed
BASE_PATH = "/Users/oscarrezab/GitHub/macalester/catkin_ws"  # definitely modify
DATASET_NAME = "train_data_annotated"  # modify as needed
DATASET_SOURCE = os.path.join(BASE_PATH, "src/collision_avoidance/res/", DATASET_NAME)
DATASET_OUTPUT = os.path.join(BASE_PATH, "src/collision_avoidance/res/", f"{DATASET_NAME}_{TASK_NAME}")
DATASET_COLAB = f"/content/{DATASET_NAME}_{TASK_NAME}/"  # the path where the dataset will live in Google Colab
LABELS_PATH = os.path.join(DATASET_SOURCE, "labels/")
IMAGES_PATH = os.path.join(DATASET_SOURCE, "images/")

# Define split ratio and randomness
TRAIN_SPLIT = 0.8
RANDOM_SEED = 480
random.seed(RANDOM_SEED)

# Define image dimensions for normalization
IMG_WIDTH = 640
IMG_HEIGHT = 480


def formatLabelLine(line: str, taskName: str, imgWidth: int, imgHeight: int) -> str:
    """
    Convert the raw label format into one that is suitable for the desired detection task.

    From:
    person_id movementStatus x0 y0 x1 y1 movement_direction height width length long_dist lat_dist

    Into:
    classId xCenter yCenter width height
    """
    # Map indices to class values for the "all" task
    all_task_dict = {"0-m": "0", "0-s": "1",
                     "90-m": "2", "90-s": "3",
                     "180-m": "4", "180-s": "5",
                     "270-m": "6", "270-s": "7"}

    direction_task_dict = {"0": "0", "90": "1", "180": "2", "270": "3"}

    attributes = line.split(" ")
    if taskName == "status":
        classId = "0" if attributes[1] == "s" else "1"  # set for movement status
    elif taskName == "direction":
        movementDirection = attributes[6]
        classId = direction_task_dict[movementDirection]  # set for movement direction
    else:  # task is "all"
        movementStatus = attributes[1]
        movementDirection = attributes[6]
        if movementStatus not in ["m", "s"]:
            return "ERROR in movement status"
        if movementDirection not in ["0", "90", "180", "270"]:
            return "ERROR in movement direction"
        combinedAttrib = f"{movementDirection}-{movementStatus}" # merge direction-status
        classId = all_task_dict[combinedAttrib]


    # Get normalized 2d-box attributes
    xCenter = (float(attributes[4]) + float(attributes[2])) / 2 / imgWidth
    width = (float(attributes[4]) - float(attributes[2])) / imgWidth
    yCenter = (float(attributes[5]) + float(attributes[3])) / 2 / imgHeight
    height = (float(attributes[5]) - float(attributes[3])) / imgHeight

    # Return the new line as string
    return f"{classId} {xCenter} {yCenter} {width} {height}"


def process_label_file(taskName: str, srcLabelPath: str, dstLabelPath: str, imgWidth: int, imgHeight: int):
    """Calls formatLabelLine for each line in the given source text file and writes the updated lines
    to the given destination text file."""
    with open(srcLabelPath, "r") as f:
        lines = f.readlines()

    processedLines = [formatLabelLine(line, taskName, imgWidth, imgHeight) for line in lines]

    if "ERROR in movement status" in processedLines:
        print(f"  ERROR: invalid movement status in file {srcLabelPath}")
    if "ERROR in movement direction" in processedLines:
        print(f"  ERROR: invalid movement direction file {srcLabelPath}")

    with open(dstLabelPath, "w") as f:
        f.write("\n".join(processedLines) + "\n")


def collectPairs(imagesPath: str, labelsPath: str) -> list:
    """Gathers all the image-label pairs."""
    pairs = []

    for root, _, files in os.walk(imagesPath):
        for file in files:
            if file.endswith(".jpg"):
                framePath = os.path.join(root, file)

                # Get corresponding label path
                relPath = os.path.relpath(framePath, imagesPath)
                labelPath = os.path.join(labelsPath, os.path.splitext(relPath)[0] + ".txt")

                if os.path.exists(labelPath):
                    pairs.append((framePath, labelPath))
                else:
                    print(f"  WARNING: Missing label for {framePath}")

    return pairs


def createDirs(outputDir: str):
    """Create output directories."""
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(outputDir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(outputDir, split, "labels"), exist_ok=True)


def splitData(pairs):
    """Split data in its directories."""
    random.shuffle(pairs)
    splitIdx = int(len(pairs) * TRAIN_SPLIT)
    return pairs[:splitIdx], pairs[splitIdx:]


def copyData(taskName: str, pairs: list, split: str, outputDir: str, imgWidth: int, imgHeight: int):
    """Copy and process data."""
    for imgPath, labelPath in pairs:
        imgName = imgPath.split("/")[-1]
        labelName = labelPath.split("/")[-1]

        dstImg = os.path.join(outputDir, split, "images", imgName)
        dstLabel = os.path.join(outputDir, split, "labels", labelName)

        shutil.copy2(imgPath, dstImg)
        process_label_file(taskName, labelPath, dstLabel, imgWidth, imgHeight)


def create_data_yaml(taskName: str, outputDir: str, dataDir: str):
    """Create the YAML file."""
    if taskName == "status":
        yamlContent = f"""
path: {dataDir}
train: train/images
val: valid/images

names:
  0: stationary
  1: moving
""".strip()
    elif taskName == "direction":
        yamlContent = f"""
path: {dataDir}
train: train/images
val: valid/images

names:
  0: 0
  1: 90
  2: 180
  3: 270
""".strip()
    else:
        yamlContent = f"""
path: {dataDir}
train: train/images
val: valid/images

names:
  0: 0-moving
  1: 0-stationary
  2: 90-moving
  3: 90-stationary
  4: 180-moving
  5: 180-stationary
  6: 270-moving
  7: 270-stationary
        """.strip()

    with open(os.path.join(outputDir, "data.yaml"), "w") as f:
        f.write(yamlContent)


if __name__ == "__main__":
    assert TASK_NAME == "status" or TASK_NAME == "direction" or TASK_NAME == "all"

    print("Collecting data...")
    collectedPairs = collectPairs(IMAGES_PATH, LABELS_PATH)
    print(f"  Found {len(collectedPairs)} image-label pairs")

    print("Creating folders...")
    createDirs(DATASET_OUTPUT)

    print("Splitting data...")
    trainPairs, validPairs = splitData(collectedPairs)

    print(f"  Train: {len(trainPairs)}, Valid: {len(validPairs)}")

    print("Copying train data...")
    copyData(TASK_NAME, trainPairs, "train", DATASET_OUTPUT, IMG_WIDTH, IMG_HEIGHT)

    print("Copying validation data...")
    copyData(TASK_NAME, validPairs, "valid", DATASET_OUTPUT, IMG_WIDTH, IMG_HEIGHT)

    print("Creating data.yaml...")
    create_data_yaml(TASK_NAME, DATASET_OUTPUT, DATASET_COLAB)

    print("Done!")
