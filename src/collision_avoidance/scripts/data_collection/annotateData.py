import cv2
import os

# Define paths, modify as needed
BASE_PATH = "/Users/oscarrezab/GitHub/macalester/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/mock_ds/")
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260416-0119frames")
IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260416-0119frames")

def getFramePaths(imagesPath: str):
    """Gets the list of paths for jpg files with frame images."""
    return [os.path.join(imagesPath, frameName) for frameName in os.listdir(imagesPath) if frameName.endswith("jpg")]

def getLabelPaths(labelsPath: str):
    """Gets the list of paths for txt files with frame annotations."""
    return[os.path.join(labelsPath, frameName) for frameName in os.listdir(labelsPath) if frameName.endswith("txt")]

def getAttributes(labelFilePath: str):
    """Returns a list of lists. The inner lists represent the attributes of a detected person.
    Attributes are ordered as:
    [personId, movementStatus, x0, y0, x1. y1, movementDirection, height, width, length,
    longitudinalDistance, latitudinalDistance]"""
    peopleAttribs = []
    with open(labelFilePath, "r") as labelsFile:
        for line in labelsFile:
            personAttribs = line.strip().split(" ")  # remove new line and separate by whitespace
            peopleAttribs.append(personAttribs)

    return peopleAttribs

def formatAttributes(personAttribs: list):
    """Put a person's attributes into a tuple of formated strings to display."""
    personId = personAttribs[0]
    movementStatus = personAttribs[1]
    x0, y0, x1, y1 = personAttribs[2], personAttribs[3], personAttribs[4], personAttribs[5]
    movementDir = personAttribs[6]
    personHeight, personWidth, personLength = personAttribs[7], personAttribs[8], personAttribs[9]
    longDistance, latDistance = personAttribs[10], personAttribs[11]

    identifiers = f"id: {personId}, movement: {movementStatus}, direction: {movementDir}"
    boundingBox = f"({x0}, {y0}), ({x1}, {y1})"
    realMeasures = f"h: {personHeight}, w: {personWidth}, l: {personLength}"
    distances = f"x: {longDistance}, d: {latDistance}"

    return identifiers, boundingBox, realMeasures, distances

def displayFrameWithAnnotations(framePath, labelPath):
    frame = cv2.imread(framePath)
    attribs = getAttributes(labelPath)[0]  # TODO: implement handling for more than one person
    ids, box, measures, distances = formatAttributes(attribs)

    # Draw the bounding box
    cv2.rectangle(frame, (int(float(attribs[2])), int(float(attribs[3]))), (int(float(attribs[4])), int(float(attribs[5]))), (0, 0, 255), 3)

    # Insert text
    cv2.putText(frame, ids, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, box, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, measures, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, distances, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Display annotated frame
    cv2.imshow("Annotated Frame", frame)

    # Handle quit
    key = cv2.waitKey()
    ch = chr(key & 0xFF)

    if ch == "q":
      cv2.destroyAllWindows()

if __name__ == "__main__":
    displayFrameWithAnnotations(framePath="/Users/oscarrezab/GitHub/macalester/catkin_ws/src/collision_avoidance/res/mock_ds/images/20260416-0119frames/frame20260416-012004.jpg",
                                labelPath="/Users/oscarrezab/GitHub/macalester/catkin_ws/src/collision_avoidance/res/mock_ds/labels/20260416-0119frames/frame20260416-012004.txt")