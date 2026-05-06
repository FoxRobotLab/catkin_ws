import cv2
import os
import numpy as np


#Adjust as needed 
BASE_PATH = "/home/ryan/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/may_3__annotated_data_apr_23/may_3__annotated_data_apr_23/annotated_data_apr_23")

IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260423-1613frames")
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260423-1613frames")

H = np.load("homography.npy")



#Basically assume when the robot is in position, the closest row of tiles will be 2 across
#to fill up the image width
X_MIN = 0.0
X_MAX = 2.0
robot_x = (X_MIN + X_MAX) / 2 

#Weird transfromation stuff from cv docs
def pixel_to_world(x, y, H):
    point = np.array([[[x, y]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, H)
    return world_point[0][0]



def classify_risk(x, y):

    dx = x - robot_x  

    base_width = 0.2
    scale = 0.15
    allowed_dx = base_width + scale * y

    #normalize lateral distance
    lateral_score = abs(dx) / allowed_dx

    #arbitrary, chosen based off of observed pattersn in images
    min_y = 4
    max_y = 5

    #use combo of lateral dist and long dist, also kind of arbitrary
    if y < min_y and lateral_score < 0.6:
        return "HIGH"

    elif y < max_y and lateral_score < 1.2:
        return "MEDIUM"

    else:
        return "LOW"


def load_bboxes(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            #Only works if we have all 4 valid bbox coords
            if len(parts) < 6:
                continue

            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            boxes.append((x1, y1, x2, y2))

    return boxes

#Just draw on boxes
def process_detections(bboxes, H):
    results = []

    for (x1, y1, x2, y2) in bboxes:
        foot_x = (x1 + x2) / 2
        foot_y = y2

        x, y = pixel_to_world(foot_x, foot_y, H)

        risk = classify_risk(x, y)

        results.append({
            "bbox": (x1, y1, x2, y2),
            "world": (float(x), float(y)),
            "risk": risk
        })

    return results


#annotate image with classifications
def draw_results(frame, results):
    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        X, Y = r["world"]
        risk = r["risk"]

        if risk == "HIGH":
            color = (0, 0, 255)
        elif risk == "MEDIUM":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      color, 2)

        label = f"{risk} ({X:.2f},{Y:.2f})"
        cv2.putText(frame, label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    return frame


#-------File Processing and frame control stuff-------

def get_frame_paths(images_path):
    return [os.path.join(images_path, f)
            for f in sorted(os.listdir(images_path))
            if f.endswith(".jpg")]


def get_label_path(image_path):
    filename = os.path.basename(image_path)
    label_name = os.path.splitext(filename)[0] + ".txt"
    return os.path.join(LABELS_PATH, label_name)


def run_viewer():
    print("\n=====================================")
    print(" Viewing risk classifications")
    print(" [n] next   [q] quit")
    print("=====================================\n")

    image_files = get_frame_paths(IMAGES_PATH)

    for image_path in image_files:
        print("Processing:", os.path.basename(image_path))

        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to load image")
            continue

        label_path = get_label_path(image_path)
        bboxes = load_bboxes(label_path)

        results = process_detections(bboxes, H)

        for r in results:
            print("World:", r["world"], "Risk:", r["risk"])

        output = draw_results(frame.copy(), results)

        while True:
            cv2.imshow("Risk Viewer", output)

            key = cv2.waitKey(0)

            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return

            elif key == ord('n') or key == ord('N'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_viewer()