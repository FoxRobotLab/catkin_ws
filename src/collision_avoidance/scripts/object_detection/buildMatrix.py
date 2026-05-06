"""--------------------------------------------------------------------------------
annotateData.py
Author: Oscar Reza B. and Ryan Maule

This program helps correct annotations for collected data with the purpose of
training a 3d object detector.

To run this program, we must have a directory with images and labels subdirectories
with the relevant jpg and txt files.

--------------------------------------------------------------------------------"""
import cv2
import os


# Define paths, modify as needed
BASE_PATH = "/home/ryan/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/callibration_frames/")
IMAGES_PATH = os.path.join(DATASET_PATH, "20260429-1833frames")

OUTPUT_IMAGES_PATH = os.path.join(DATASET_PATH, "annotated_images")
os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)


#interface will be to click into an image and click to add a point to the screen
#that point will get stored in a matrix data struct
#Then click next to move to next image


#used fot manually drawing bounding box
drawing = False
ix, iy = -1, -1
new_box = None


points = []

def draw_point(event, x, y, tags, param):
    """Click on a point where an edge is"""
    global drawing, ix, iy, new_box

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y


def getFramePaths(imagesPath: str):
    """Gets the list of paths for jpg files with frame images."""
    return [os.path.join(imagesPath, frameName) for frameName in sorted(os.listdir(imagesPath)) if frameName.endswith("jpg")]

def drawGrid(framePath):
    print("drawing grid")
    frame = cv2.imread(framePath).copy()
    #Will basically take an image and draw the points on it 
    for point in points:
        x = point.split(",")[0].strip()
        y = point.split(",")[1].strip()
        cv2.circle(frame, (int(x), int(y)), radius=3, color=(0,0,255), thickness=-1)

    out_path = os.path.join(OUTPUT_IMAGES_PATH, os.path.basename(framePath))
    cv2.imwrite(out_path, frame)



def buildStruct(framePath):
    with open('points2.txt', encoding="utf-8") as f:
        for line in f:
            points.append(line)
    drawGrid(framePath)

def displayAndModify(framePath):
    """Displays the given frame and the label for an identified person.
    Waits for user input on the command line."""

    cv2.namedWindow("Annotated Frame")
    cv2.setMouseCallback("Annotated Frame", draw_point)

    global new_box


    while True:
        frame = cv2.imread(framePath).copy()

        color = (0, 0, 255)



        cv2.putText(frame,
                    "[n] next [b] add point  [q] quit",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("Annotated Frame", frame)

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

        #Draw New bounding box
        elif key == ord('b'):
            print("Draw point with mouse...")

            while True:
                temp_frame = frame.copy()

                cv2.circle(temp_frame, (ix, iy), radius=3, color=color, thickness=-1)

                cv2.imshow("Annotated Frame", temp_frame)
                k = cv2.waitKey(1)

                #On enter, box gets confirmed
                if k == 13:
                    x, y = ix, iy


                    print(f"Added Point: {x}, {y}")
                    points.append((x, y))
                    with open("points2.txt", "a") as f:
                        new_line = f"{x},{y}\n"
                        f.write(new_line)
                    break

                #On esc, cancel point drawing
                elif k == 27:
                    print("Cancelled")
                    break


def runModifier(imagesPath):
    """Looks over all frames in the given paths and runs the displayAndModify() method.
    This is the main function of this program. To quit, simply type 'q' and hit enter."""
    print("\n=====================================================")
    print(" Modifying annotations. Type 'q' to exit at any time")
    print("=====================================================\n")

    imagesList = sorted(getFramePaths(imagesPath))

    for imageFile in imagesList:
        print("Processing frame:", os.path.basename(imageFile))
        displayAndModify(imageFile)


if __name__ == "__main__":
    #buildStruct("/home/ryan/catkin_ws/src/collision_avoidance/res/callibration_frames/20260429-1811frames/frame20260429-181224.jpg")
    runModifier(IMAGES_PATH)