import cv2
import os
import math

# Define paths, modify as needed
BASE_PATH = "/home/ryan/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/mock_ds/")
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260416-0119frames")
IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260416-0119frames")

#output folder for annotated images
OUTPUT_PATH = os.path.join(DATASET_PATH, "annotated_images")
os.makedirs(OUTPUT_PATH, exist_ok=True)

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

#estimate 3D position from 2D bounding box
def estimate_3d_position_from_bbox(x0, y0, x1, y1, img_w, real_height=1.75):
    """
    Takes in known dimensions and extrapilates into 3D bounding box
    """

    #use bottom of bbox as anchor reference
    bbox_h = max(1e-6, (y1 - y0))

    bbox_cx = (x0 + x1) / 2

    f = img_w
    cx_img = img_w / 2

    #stabilize depth using bottom anchor assumption
    z = (f * real_height) / bbox_h

    # lateral position unchanged
    x = (bbox_cx - cx_img) * z / f

    return x, z

#match label file by image filename
def getMatchingLabelPath(imagePath):
    imageName = os.path.basename(imagePath)
    labelName = imageName.replace(".jpg", ".txt")
    return os.path.join(LABELS_PATH, labelName)

def project_point(x,y,z, f, cx, cy):
    """Project 3D point (camera coords) to 2D image."""
    if z <= 0.1:  # avoid division issues
        return None
    x = f * (x / z) + cx
    y = f * (-y / z) + cy  # negative because image y goes down
    return int(x), int(y)


def get_3d_box_corners(cx, cy, cz, w, h, l):
    """Return 8 corners of 3D bounding box."""
    x_offsets = [-w/2, w/2]
    y_offsets = [-h/2, h/2]  # ground to top
    z_offsets = [-l/2, l/2]

    corners = []
    for x in x_offsets:
        for y in y_offsets:
            for z in z_offsets:
                corners.append((
                    cx + x,
                    cy + y,
                    cz + z
                ))
    return corners


def draw_3d_box(frame, corners_2d):
    """Draw lines between projected 3D box corners."""

    # Define edges into corners list
    edges = [
        (0,1),(1,3),(3,2),(2,0),  # bottom
        (4,5),(5,7),(7,6),(6,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # verticals
    ]

    for i, j in edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            cv2.line(frame, corners_2d[i], corners_2d[j], (0,255,0), 2)


def displayFrameWithAnnotations(framePath, labelPath):
    frame = cv2.imread(framePath)
    h_img, w_img = frame.shape[:2]

    people = getAttributes(labelPath)

    #handle empty label files (no people in frame)
    if len(people) == 0:
        return frame  # just return clean image with no annotations

    attribs = people[0]
    ids, box, measures, distances = formatAttributes(attribs)

    # Parse values
    x0, y0, x1, y1 = map(float, attribs[2:6])

    #compute x and z from 2D bounding box 
    cx, cz = estimate_3d_position_from_bbox(x0, y0, x1, y1, w_img, h_img)

    cy = 0   

    h = float(attribs[7])
    w = float(attribs[8])
    l = float(attribs[9])

    # Camera intrinsics (approx)
    f = w_img  # simple approximation
    cx_img = w_img / 2
    cy_img = h_img / 2

    # Build 3D box
    corners_3d = get_3d_box_corners(cx, cy, cz, w, h, l)

    # Project to 2D
    corners_2d = [
        project_point(x, y, z, f, cx_img, cy_img)
        for (x, y, z) in corners_3d
    ]

    # Draw 3D box
    draw_3d_box(frame, corners_2d)

    #Insert text
    cv2.putText(frame, ids, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, measures, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, distances, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame  


#batch processing pipeline
def runBatch():
    framePaths = sorted(getFramePaths(IMAGES_PATH))

    for framePath in framePaths:

        labelPath = getMatchingLabelPath(framePath)

        if not os.path.exists(labelPath):
            print(f"Missing label for {framePath}, skipping")
            continue

        frame = displayFrameWithAnnotations(framePath, labelPath)

        outName = os.path.basename(framePath)
        outPath = os.path.join(OUTPUT_PATH, outName)

        cv2.imwrite(outPath, frame)

        print(f"Saved: {outPath}")


if __name__ == "__main__":
    runBatch()