import cv2
import os
import numpy as np


BASE_PATH = "/home/ryan/catkin_ws"
DATASET_PATH = os.path.join(BASE_PATH, "src/collision_avoidance/res/may_3__annotated_data_apr_23/may_3__annotated_data_apr_23/annotated_data_apr_23")

IMAGES_PATH = os.path.join(DATASET_PATH, "images/20260423-1613frames")
LABELS_PATH = os.path.join(DATASET_PATH, "labels/20260423-1613frames")

OUTPUT_PATH = os.path.join(DATASET_PATH, "risk_annotated_frames")
os.makedirs(OUTPUT_PATH, exist_ok=True)

H = np.load("homography.npy")

X_MIN = 0.0
X_MAX = 2.0
robot_x = (X_MIN + X_MAX) / 2

#Got this from documentatino, but maps x,y coords to real world points
#based on homoegraphy file
def pixel_to_world(x, y, H):
    point = np.array([[[x, y]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, H)
    return world_point[0][0]

#Check rule table for assumptions and rule judgement
def classify_risk(x, y, direction, movementStatus):

    dx = x - robot_x


    #if person x is higher then robot x they are to the right 
    right = dx > 0

    min_y = 3.75
    max_y = 4.5

    #Create a function to bump up just one risk level

    print("dx: ", abs(dx))
    if y < min_y and abs(dx) < 0.6:

        #This will be classified as high unless person is moving away from bot
        if movementStatus.lower() == 'm':
            
            #If person is moving away from robot 
            if right and direction == '270':
                return "LOW"
            
            if not right and direction == '90':
                return "LOW"
            
            #If person is moving away from robot longitudally
            if direction == '0':
                return "LOW"

        return "HIGH"

    elif y < max_y:
        #handle movement status
        if movementStatus.lower() == 'm' and abs(dx) < 1.2:

            #For medium if a person is facing the bot they could be classified as higher risk
            if direction == '180':
                return "HIGH"
            
            #If person is moving away from robot longitudally
            if direction == '0':
                return "LOW"

            #If person is moving away from robot laterally
            if right and direction == '270':
                return "LOW"
            
            if not right and direction == '90':
                return "LOW"


        return "MEDIUM"

    else:

        #For now dont worry about movement status for low 
        return "LOW"


def load_data(label_path):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
            direction = parts[6]
            movementStatus = parts[1]

            boxes.append((x1, y1, x2, y2, direction, movementStatus))

    return boxes


def process_detections(data, H):
    results = []

    risk_priority = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2
    }


    for (x1, y1, x2, y2, direction, movementStatus) in data:

        width = x2 - x1

        sample_points = [
            x1 + 0.15 * width,
            (x1 + x2) / 2,
            x2 - 0.15 * width
        ]

        highest_risk = "LOW"

        center_x = (x1 + x2) / 2
        best_world = pixel_to_world(center_x, y2, H)

        for px in sample_points:

            world_x, world_y = pixel_to_world(px, y2, H)

            current_risk = classify_risk(world_x, world_y, direction, movementStatus)

            if risk_priority[current_risk] > risk_priority[highest_risk]:
                highest_risk = current_risk
                best_world = (world_x, world_y)

        results.append({
            "bbox": (x1, y1, x2, y2),
            "world": (float(best_world[0]), float(best_world[1])),
            "risk": highest_risk
        })

    return results


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

        cv2.putText(frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)

    return frame


def save_annotated_frame(output_frame, image_path):

    filename = os.path.basename(image_path)

    out_path = os.path.join(OUTPUT_PATH, filename)

    cv2.imwrite(out_path, output_frame)

    print("Saved:", out_path)


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

        data = load_data(label_path)

        results = process_detections(data, H)

        for r in results:
            print("World:", r["world"], "Risk:", r["risk"])

        output = draw_results(frame.copy(), results)

        save_annotated_frame(output, image_path)

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