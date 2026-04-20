import os

# Define paths, modify as needed
LABELS_FOLDER = "/home/ryan/catkin_ws/src/collision_avoidance/res/mock_ds_unann/labels/20260416-0119frames"
OUTPUT_FOLDER = os.path.join(LABELS_FOLDER, "converted")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def estimate(x0, y0, x1, y1, img_w=640, real_h=1.75):

    # bbox geometry
    bbox_h = max(1e-6, y1 - y0)
    cx = (x0 + x1) / 2
    cx_img = img_w / 2

    # depth estimate
    z = (img_w * real_h) / bbox_h

    # lateral estimate (scaled by depth)
    x = -((cx - cx_img) * z / img_w)

    return round(x, 2), round(z, 2)

#Open and read in data, then replace the placeholders with actual values
def process_file(in_path, out_path):
    with open(in_path, "r") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 12:
            continue

        x0 = float(parts[2])
        y0 = float(parts[3])
        x1 = float(parts[4])
        y1 = float(parts[5])

        latDist, longDist = estimate(x0, y0, x1, y1)

        # replace placeholders
        parts[-2] = str(longDist)
        parts[-1] = str(latDist)

        new_lines.append(" ".join(parts))

    with open(out_path, "w") as f:
        f.write("\n".join(new_lines))


#Run for each file in folder
def run_batch():
    for file_name in os.listdir(LABELS_FOLDER):
        if not file_name.endswith(".txt"):
            continue

        in_path = os.path.join(LABELS_FOLDER, file_name)
        out_path = os.path.join(OUTPUT_FOLDER, file_name)

        process_file(in_path, out_path)

        print(f"Processed: {file_name}")


if __name__ == "__main__":
    run_batch()