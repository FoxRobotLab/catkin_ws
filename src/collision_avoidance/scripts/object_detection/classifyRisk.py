"""
A class to classify risk of collision based on the predicted bounding box, movement status, and movement direction.

To do this, we first compute the estimate real-world position of each pedestrian using the predicted bounding box,
then we add the other attributes to determine the risk of collision.

Authors: Ryan Maule and Oscar Reza B.
Spring 2026
"""
import cv2
import os
import numpy as np


class RiskClassifier:
    def __init__(self, base_path, save_flag=False, homography_path="homography.npy", verbose=False):
        self.SAVE_FLAG = save_flag
        self.BASE_PATH = base_path
        self.DATASET_PATH = os.path.join(self.BASE_PATH, "src/collision_avoidance/res/train_data_annotated")
        self.IMAGES_PATH = os.path.join(self.DATASET_PATH, "images/20260423-1613frames")
        self.LABELS_PATH = os.path.join(self.DATASET_PATH, "labels/20260423-1613frames")
        self.OUTPUT_PATH = os.path.join(self.DATASET_PATH, "risk_annotated_frames")
        self.VERBOSE = verbose

        os.makedirs(self.OUTPUT_PATH, exist_ok=True)

        self.H = np.load(homography_path)

        self.X_MIN = 0.0
        self.X_MAX = 2.0
        self.robot_x = (self.X_MIN + self.X_MAX) / 2

        self.risk_priority = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2
        }

    def pixel_to_world(self, x, y):
        """
        Maps image pixel coordinates to real-world coordinates using the homography matrix.
        """
        point = np.array([[[x, y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.H)

        return world_point[0][0]

    def classify_risk(self, x, y, direction, movement_status):
        """
        Rule-based risk classifier.
        """
        dx = x - self.robot_x

        # If person x is higher than robot x they are to the right
        right = dx > 0

        min_y = 3.75
        max_y = 4.5

        if self.VERBOSE:
            print("dx:", abs(dx))

        # HIGH RISK ZONE
        if y < min_y and abs(dx) < 0.6:
            if movement_status.lower() == 'm':
                # Moving away laterally
                if right and direction == '270':
                    return "LOW"

                if not right and direction == '90':
                    return "LOW"

                # Moving away longitudinally
                if direction == '0':
                    return "LOW"

            return "HIGH"

        # MEDIUM RISK ZONE
        elif y < max_y:
            if movement_status.lower() == 'm' and abs(dx) < 1.2:
                # Facing robot
                if direction == '180':
                    return "HIGH"

                # Moving away longitudinally
                if direction == '0':
                    return "LOW"

                # Moving away laterally
                if right and direction == '270':
                    return "LOW"

                if not right and direction == '90':
                    return "LOW"

            return "MEDIUM"

        # LOW RISK ZONE
        else:
            return "LOW"

    def load_data(self, label_path):
        boxes = []

        if not os.path.exists(label_path):
            return boxes

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 7:
                    continue

                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])

                direction = parts[6]
                movement_status = parts[1]

                boxes.append((
                    x1,
                    y1,
                    x2,
                    y2,
                    direction,
                    movement_status
                ))

        return boxes

    def process_detections(self, data):
        results = []

        for (x1, y1, x2, y2, direction, movement_status) in data:
            width = x2 - x1
            sample_points = [
                x1 + 0.15 * width,
                (x1 + x2) / 2,
                x2 - 0.15 * width
            ]
            highest_risk = "LOW"

            center_x = (x1 + x2) / 2
            best_world = self.pixel_to_world(center_x, y2)

            for px in sample_points:
                world_x, world_y = self.pixel_to_world(px, y2)

                current_risk = self.classify_risk(
                    world_x,
                    world_y,
                    direction,
                    movement_status
                )

                if self.risk_priority[current_risk] > self.risk_priority[highest_risk]:
                    highest_risk = current_risk
                    best_world = (world_x, world_y)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "world": (
                    float(best_world[0]),
                    float(best_world[1])
                ),
                "risk": highest_risk
            })

        return results

    def draw_results(self, frame, results):
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

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            label = f"{risk} ({X:.2f},{Y:.2f})"

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return frame

    def save_annotated_frame(self, output_frame, image_path):
        filename = os.path.basename(image_path)
        out_path = os.path.join(self.OUTPUT_PATH, filename)

        cv2.imwrite(out_path, output_frame)
        print("Saved:", out_path)

    def get_frame_paths(self):
        return [
            os.path.join(self.IMAGES_PATH, f)
            for f in sorted(os.listdir(self.IMAGES_PATH))
            if f.endswith(".jpg")
        ]

    def get_label_path(self, image_path):
        filename = os.path.basename(image_path)
        label_name = os.path.splitext(filename)[0] + ".txt"

        return os.path.join(self.LABELS_PATH, label_name)

    def run_viewer(self):

        print("\n=====================================")
        print(" Viewing risk classifications")
        print(" [n] next   [q] quit")
        print("=====================================\n")

        image_files = self.get_frame_paths()

        for image_path in image_files:
            print("Processing:", os.path.basename(image_path))

            frame = cv2.imread(image_path)
            if frame is None:
                print("Failed to load image")
                continue

            label_path = self.get_label_path(image_path)
            data = self.load_data(label_path)
            results = self.process_detections(data)

            for r in results:
                print(
                    "World:",
                    r["world"],
                    "Risk:",
                    r["risk"]
                )

            output = self.draw_results(frame.copy(), results)

            if self.SAVE_FLAG:
                self.save_annotated_frame(output, image_path)

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
    classifier = RiskClassifier(
        base_path="/Users/oscarrezab/GitHub/macalester/catkin_ws",
        save_flag=False,
        homography_path="homography.npy"
    )

    classifier.run_viewer()