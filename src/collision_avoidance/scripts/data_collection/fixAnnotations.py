"""--------------------------------------------------------------------------------
annotateData.py
Author: Oscar Reza B. and Ryan Maule

This program helps correct annotations for collected data with the purpose of
training a 3d object detector.

To run this program, we must have a directory with images and labels subdirectories
with the relevant jpg and txt files.

TODO: this program currently works only for frames with one person in it.
    We must modify it to handle multiple people visible
--------------------------------------------------------------------------------"""
import cv2
import os

# Define paths, modify as needed
BASE_PATH = "/Users/oscarrezab/GitHub/macalester/catkin_ws"  # definitely modify
DATASET_NAME = "mock_ds"  # modify as needed
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/", DATASET_NAME)
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260416-0119frames")  # modify as needed
IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260416-0119frames")  # modify as needed

def getFramePaths(imagesPath: str):
    """Gets the list of paths for jpg files with frame images."""
    return [os.path.join(imagesPath, frameName) for frameName in sorted(os.listdir(imagesPath)) if frameName.endswith("jpg")]

def getLabelPaths(labelsPath: str):
    """Gets the list of paths for txt files with frame annotations."""
    return[os.path.join(labelsPath, frameName) for frameName in sorted(os.listdir(labelsPath)) if frameName.endswith("txt")]

def getAttributes(labelFilePath: str):
    """Returns a list of lists. The inner lists represent the attributes of a detected person.
    Attributes are ordered as:
    [personId, movementStatus, x0, y0, x1. y1, movementDirection, height, width, length,
    longitudinalDistance, latitudinalDistance]"""
    peopleAttribs = []
    with open(labelFilePath, "r") as labelsFile:
        for line in labelsFile:
            if line == "":
                return peopleAttribs
            personAttribs = line.strip().split(" ")  # remove new line and separate by whitespace
            peopleAttribs.append(personAttribs)

    return peopleAttribs

def formatAttributes(personAttribs: list):
    """Put a person's attributes into a tuple of formated strings to display."""
    if len(personAttribs) == 0:
        return "", "", "", ""
    personId = personAttribs[0]
    movementStatus = personAttribs[1]
    x0, y0, x1, y1 = personAttribs[2], personAttribs[3], personAttribs[4], personAttribs[5]
    movementDir = personAttribs[6]
    personHeight, personWidth, personLength = personAttribs[7], personAttribs[8], personAttribs[9]
    longDistance, latDistance = personAttribs[10], personAttribs[11]

    identifiers = f"id: {personId}, movement: {movementStatus}, direction: {movementDir}"
    boundingBox = f"({x0}, {y0}), ({x1}, {y1})"
    realMeasures = f"h: {personHeight}, w: {personWidth}, l: {personLength}"
    distances = f"d: {longDistance}, x: {latDistance}"

    return identifiers, boundingBox, realMeasures, distances


def updateAnnotations(labelPath, movementStatus, boxCoords, movementDir, longitudinalDist, latitudinalDist):
    """Grabs the first line and updates it with the given attributes."""
    # Get current label
    with open(labelPath, 'r') as file:
        lines = file.readlines()

    # TODO: implement to handle multiple people in a single frame
    currentAnnot = "- - - - - - - - - - - -"
    if len(lines) > 0:
        currentAnnot = lines[0]

    # Decompose current annotations
    annotAttribs = currentAnnot.split(" ")

    # Update label attributes, handling for blank inputs
    annotAttribs[1] = annotAttribs[1] if movementStatus == "" else movementStatus
    annotAttribs[6] = annotAttribs[6] if movementDir == "" else movementDir
    annotAttribs[10] = annotAttribs[10] if longitudinalDist == "" else longitudinalDist
    annotAttribs[11] = annotAttribs[11] if latitudinalDist == "" else latitudinalDist

    # Handle bounding box coordinates separately
    if boxCoords != "":
        x0, y0, x1, y1 = boxCoords.split(" ")
        annotAttribs[2] = x0
        annotAttribs[3] = y0
        annotAttribs[4] = x1
        annotAttribs[5] = y1

    # Get new annotation format and replace
    newAnnot = f"{annotAttribs[0]} {annotAttribs[1]} {annotAttribs[2]} {annotAttribs[3]} {annotAttribs[4]} {annotAttribs[5]} {annotAttribs[6]} {annotAttribs[7]} {annotAttribs[8]} {annotAttribs[9]} {annotAttribs[10]} {annotAttribs[11]}"
    if len(lines) > 0:
        lines[0] = newAnnot  # TODO: modify to handle multiple people in one frame
    else:
        lines.append(newAnnot)

    # Write all lines back to the file
    with open(labelPath, 'w') as file:
        file.writelines(lines)


def userCLI(framePath, labelPath):
    """Displays the current frame to visualize, asks the user about modifying the annotations for
    the given frame, handles user input, and calls the `updateAnnotations()` method."""
    frameName = framePath.split("/").pop()
    print(f"Visualizing {frameName}")
    flag = input("  Would you like to modify these annotations? (Y/n): ")
    if flag.lower() == "y":
        print("  ! Leave any field blank to keep current value")
        movementStatus = input("  -> movementStatus as \"s\" or \"m\": ")
        boxCoords = input("  -> boundingBoxCoords as \"x0 y0 x1 y1\": ")
        movementDir = input("  -> movementDir as 0, 90, 180, or 270: ")
        longitudinalDist = input("  -> longitudinalDist in meters: ")
        latitudinalDist = input("  -> latitudinalDist in meters: ")

        # Update the annotation
        updateAnnotations(labelPath, movementStatus, boxCoords, movementDir,
                          longitudinalDist, latitudinalDist)

        return "y"
    elif flag.lower() == "n":
        print("    Got it, moving to next file")
        return "n"
    elif flag.lower() == "q":
        print("\n=====================================================")
        print("Got it, exiting program...")
        exit(0)
    else:
        print("ERROR: Please enter 'y' to modify annotations or 'n' to not modify them.")
        return userCLI(framePath, labelPath)


def displayAndModify(framePath, labelPath):
    """Displays the given frame and the label for an identified person.
    Waits for user input on the command line."""
    while True:
        frame = cv2.imread(framePath)
        attribs = getAttributes(labelPath)
        if len(attribs) != 0:
            firstPersonAttribs = attribs[0]  # TODO: implement handling for more than one person
            ids, box, measures, distances = formatAttributes(firstPersonAttribs)
            # Draw the bounding box
            cv2.rectangle(frame, (int(float(firstPersonAttribs[2])), int(float(firstPersonAttribs[3]))),
                          (int(float(firstPersonAttribs[4])), int(float(firstPersonAttribs[5]))), (0, 0, 255), 3)
        else:
            ids, box, measures, distances = "ID not found", "Box not found", "Measures not found", "Distances not found"

        # Insert text
        cv2.putText(frame, ids, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, box, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, measures, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, distances, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Display annotated frame
        cv2.imshow("Annotated Frame", frame)

        # Handle quit
        key = cv2.waitKey(1)
        ch = chr(key & 0xFF)

        if ch == "q":
            cv2.destroyAllWindows()
            break

        # Handle CLI
        cliResponse = userCLI(framePath, labelPath)
        if cliResponse == "n":
            break

def runModifier(imagesPath, labelsPath):
    """Looks over all frames in the given paths and runs the displayAndModify() method.
    This is the main function of this program. To quit, simply type 'q' and hit enter."""
    # Print message for CLI
    print("\n=====================================================")
    print(" Modifying annotations. Type 'q' to exit at any time")
    print("=====================================================\n")

    # Get the file names of all images and labels
    imagesList = sorted(getFramePaths(imagesPath))
    labelsList = sorted(getLabelPaths(labelsPath))

    # Display and modify every frame in the list, one by one
    for imageFile, labelFile in zip(imagesList, labelsList):
        displayAndModify(imageFile, labelFile)


if __name__ == "__main__":
    runModifier(IMAGES_PATH, LABELS_PATH)
