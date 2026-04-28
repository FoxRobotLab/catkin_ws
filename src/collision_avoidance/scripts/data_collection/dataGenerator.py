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

# Define task name ("status" or "direction"), modify as needed
TASK_NAME = "direction"

# Define paths, modify as needed
BASE_PATH = "/Users/oscarrezab/GitHub/macalester/catkin_ws"  # definitely modify
DATASET_NAME = "mock_ds"  # modify as needed
DATASET_SOURCE = os.path.join(BASE_PATH, "src/collision_avoidance/res/", DATASET_NAME)
DATASET_OUTPUT = os.path.join(BASE_PATH, "src/collision_avoidance/res/", f"{DATASET_NAME}_{TASK_NAME}")
LABELS_PATH = os.path.join(DATASET_SOURCE, "labels/20260416-0119frames")  # modify as needed
IMAGES_PATH = os.path.join(DATASET_SOURCE, "images/20260416-0119frames")  # modify as needed

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

    attributes = line.split(" ")
    if taskName == "status":
        classId = "0" if attributes[1] == "s" else "1"  # set for movement status
    else:
        classId = attributes[6]  # set for movement direction

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

    with open(dstLabelPath, "w") as f:
        f.write("\n".join(processedLines) + "\n")


def collectPairs(imagesPath: str, labelsPath: str) -> list:
    """Gathers all the image-label pairs."""
    pairs = []

    framePaths = [os.path.join(imagesPath, frameName) for frameName in sorted(os.listdir(imagesPath)) if frameName.endswith("jpg")]
    labelPaths = [os.path.join(labelsPath, frameName) for frameName in sorted(os.listdir(labelsPath)) if frameName.endswith("txt")]

    for framePath, labelPath in zip(framePaths, labelPaths):
        pairs.append((framePath, labelPath))

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


def create_data_yaml(taskName: str, outputDir: str):
    """Create the YAML file."""
    if taskName == "status":
        yamlContent = f"""
path: {outputDir}
train: train/images
val: valid/images

names:
  0: standing
  1: walking
""".strip()
    else:
        yamlContent = f"""
path: {outputDir}
train: train/images
val: valid/images

names:
  0: 0
  90: 90
  180: 180
  270: 270
""".strip()

    with open(os.path.join(outputDir, "data.yaml"), "w") as f:
        f.write(yamlContent)


if __name__ == "__main__":
    assert TASK_NAME == "status" or TASK_NAME == "direction"

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
    create_data_yaml(TASK_NAME, DATASET_OUTPUT)

    print("Done!")
