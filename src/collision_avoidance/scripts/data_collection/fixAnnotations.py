"""--------------------------------------------------------------------------------
fixAnnotations.py
Authors: Oscar Reza B. and Ryan Maule
Spring 2026

This program helps correct annotations for collected data with the purpose of
training a 3d object detector. It can handle multiple people in a single frame.

To run this program, ensure all txt files have a new empty line at the end for the
script to work properly.

Usage: modify the paths to look into the desired folders. Then use the cv2 windows
and the command line to modify the annotations as needed,

TODO: add an assertion to check when there are no corresponding txt files for a
    given image file. That is, throw a warning or exception if there are more
    jpg files than txt files in a given directory, and viceversa.
--------------------------------------------------------------------------------"""
import cv2
import os
import numpy as np

# Define paths, modify as needed
BASE_PATH = "/Users/oscarrezab/GitHub/macalester/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/annotated_data_apr_23/")
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260423-####frames")
IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260423-####frames")

# Flag for saving annotated frames
OUTPUT_FLAG = False  # set as desired
if OUTPUT_FLAG:
    OUTPUT_IMAGES_PATH = os.path.join(DATASET_PATH, "annotated_images")
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

# Used fot manually drawing bounding box
drawing = False
ix, iy = -1, -1
new_box = None

def draw_bbox(event, x, y, tags, param):
    """When YOLO misses a person, type b to draw a new bounding box, then hit enter to
    confirm the bounding box."""
    global drawing, ix, iy, new_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            new_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        new_box = (ix, iy, x, y)


def addNewPerson(labelPath, boxCoords, personId):
    """After confirming new bounding box for a person, add with default vals to labels file."""
    x0, y0, x1, y1 = boxCoords
    new_line = f"{personId} movementStatus {x0} {y0} {x1} {y1} theta 1.75 0.5 0.5 longDistInM latDistInM\n"
    with open(labelPath, "a") as f:
        f.write(new_line)


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
            personAttribs = line.strip().split(" ")
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


def updateAnnotations(labelPath, personIndex, movementStatus, boxCoords, movementDir, height, width, longitudinalDist="longitudinalDist", latitudinalDist="latitudinalDist"):
    """Grabs the first line and updates it with the given attributes."""
    with open(labelPath, 'r') as file:
        lines = file.readlines()

    currentAnnot = "- - - - - - - - - - - -"
    if len(lines) > 0:
        line = lines[personIndex]
        currentAnnot = line
        annotAttribs = currentAnnot.split(" ")

        annotAttribs[1] = annotAttribs[1] if movementStatus == "" else movementStatus
        annotAttribs[6] = annotAttribs[6] if movementDir == "" else movementDir
        annotAttribs[7] = annotAttribs[7] if height == "" else height
        annotAttribs[8] = annotAttribs[8] if width == "" else width
        annotAttribs[10] = annotAttribs[10] if longitudinalDist == "" else longitudinalDist
        annotAttribs[11] = annotAttribs[11] if latitudinalDist == "" else latitudinalDist

        if boxCoords != "":
            x0, y0, x1, y1 = boxCoords.split(" ")
            annotAttribs[2] = x0
            annotAttribs[3] = y0
            annotAttribs[4] = x1
            annotAttribs[5] = y1

        newAnnot = f"{annotAttribs[0]} {annotAttribs[1]} {annotAttribs[2]} {annotAttribs[3]} {annotAttribs[4]} {annotAttribs[5]} {annotAttribs[6]} {annotAttribs[7]} {annotAttribs[8]} {annotAttribs[9]} {annotAttribs[10]} {annotAttribs[11]}\n"
        if len(lines) > 0:
            lines[personIndex] = newAnnot
        else:
            lines.append(newAnnot)

        with open(labelPath, 'w') as file:
            file.writelines(lines)


def displayAndModify(framePath, labelPath):
    """Displays the given frame and the label for an identified person.
    Waits for user input on the command line."""

    cv2.namedWindow("Annotated Frame")
    cv2.setMouseCallback("Annotated Frame", draw_bbox)

    personIndex = 0
    global new_box


    while True:
        frame = cv2.imread(framePath).copy()
        attribs = getAttributes(labelPath)
        attribs_window = np.zeros((360, 640, 3), dtype=np.uint8)

        if len(attribs) != 0:
            colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 255, 0)]

            for i in range(0, len(attribs)):
                person = attribs[i]
                color = colors[i % len(colors)]

                x0 = int(float(person[2]))
                y0 = int(float(person[3]))
                x1 = int(float(person[4]))
                y1 = int(float(person[5]))

                thickness = 5 if i == personIndex else 2

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)

                label_y = max(15, y0 - 10)
                cv2.putText(frame, f"ID: {person[0]}",
                            (x0, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2)

                ids, box, measures, distances = formatAttributes(person)

                cv2.putText(attribs_window, ids, (10, i*40 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(attribs_window, box, (320, i*40 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(attribs_window, measures, (10, i*40 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(attribs_window, distances, (320, i*40 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        else:
            ids, box, measures, distances = "No people", "", "", ""

        if new_box is not None:
            x0, y0, x1, y1 = new_box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)



        cv2.putText(attribs_window,
                    "[n] next  [m] modify  [j/k] switch person  [b] add box  [q] quit",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Annotated Frame", frame)
        cv2.imshow("People information", attribs_window)

        if OUTPUT_FLAG:
            out_path = os.path.join(OUTPUT_IMAGES_PATH, os.path.basename(framePath))
            cv2.imwrite(out_path, frame)

        key = cv2.waitKey(0)

        #quit
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            exit(0)

        #next image
        elif key == ord('n') or key == ord('N'):
            new_box = None
            break

        #Move to next person
        elif key == ord('j'):
            if len(attribs) > 0:
                personIndex = (personIndex + 1) % len(attribs)

        #Move to prev person
        elif key == ord('k'):
            if len(attribs) > 0:
                personIndex = (personIndex - 1) % len(attribs)

        #Triger cli for modifications
        elif key == ord('m') and len(attribs) > 0:

            print("\nModify person", personIndex + 1)
            print("Leave blank to keep value")

            movementStatus = input("movementStatus (s/m): ")
            boxCoords = input("bbox (x0 y0 x1 y1): ")
            movementDir = input("direction (0/90/180/270): ")
            height = input("height: ")
            width = input("width: ")
            # longitudinalDist = input("long dist: ")
            # latitudinalDist = input("lat dist: ")
            print(f"\n\tdone with this image person {personIndex + 1}\n")

            updateAnnotations(labelPath,
                              personIndex,
                              movementStatus,
                              boxCoords,
                              movementDir,
                              height,
                              width,
                              "longitudinalDist",
                              "latitudinalDist")

        #Draw New bounding box
        elif key == ord('b'):
            print("Draw bounding box with mouse...")

            new_box = None

            while True:
                temp_frame = frame.copy()

                if new_box is not None:
                    x0, y0, x1, y1 = new_box
                    cv2.rectangle(temp_frame, (x0, y0), (x1, y1), (0, 0, 255), 2)

                cv2.imshow("Annotated Frame", temp_frame)
                k = cv2.waitKey(1)

                #On enter, box gets confirmed
                if k == 13 and new_box is not None:
                    x0, y0, x1, y1 = new_box

                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)

                    attribs = getAttributes(labelPath)
                    new_id = len(attribs) + 1

                    addNewPerson(labelPath, (x0, y0, x1, y1), new_id)

                    print(f"Added person {new_id}")
                    break

                #On esc, cancel bounding box drawing
                elif k == 27:
                    print("Cancelled")
                    break


def runModifier(imagesPath, labelsPath):
    """Looks over all frames in the given paths and runs the displayAndModify() method.
    This is the main function of this program. To quit, simply type 'q' and hit enter."""
    print("\n=====================================================")
    print(" Modifying annotations. Type 'q' to exit at any time")
    print("=====================================================\n")

    imagesList = sorted(getFramePaths(imagesPath))
    labelsList = sorted(getLabelPaths(labelsPath))

    for imageFile, labelFile in zip(imagesList, labelsList):
        print("Processing frame:", os.path.basename(imageFile))
        displayAndModify(imageFile, labelFile)


def renderFromLabels(imagesPath, labelsPath, outputPath):
    """Can be used to annotate the images automatically from labels"""
    imagesList = sorted(getFramePaths(imagesPath))
    labelsList = sorted(getLabelPaths(labelsPath))

    os.makedirs(outputPath, exist_ok=True)

    print("\n=====================================================")
    print(" RENDERING FROM LABELS ONLY")
    print("=====================================================\n")

    for imageFile, labelFile in zip(imagesList, labelsList):

        frame = cv2.imread(imageFile)
        if frame is None:
            continue

        attribs = getAttributes(labelFile)

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        for i, person in enumerate(attribs):

            color = colors[i % len(colors)]

            try:
                x0 = int(float(person[2]))
                y0 = int(float(person[3]))
                x1 = int(float(person[4]))
                y1 = int(float(person[5]))
            except:
                continue

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

            person_id = person[0]
            cv2.putText(frame,
                        f"ID: {person_id}",
                        (x0, max(15, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

        out_file = os.path.join(outputPath, os.path.basename(imageFile))
        cv2.imwrite(out_file, frame)

        print("Rendered:", os.path.basename(imageFile))

if __name__ == "__main__":
    # out_dir = os.path.join(DATASET_PATH, "rendered_from_labels")
    # renderFromLabels(IMAGES_PATH, LABELS_PATH, out_dir)
    runModifier(IMAGES_PATH, LABELS_PATH)