import os

LABELS_FOLDER = "/home/ryan/catkin_ws/src/collision_avoidance/res/data_collection_apr_23/labels/20260423-1557frames"


def estimate(x0, y0, x1, y1, real_h, img_w=640):

    bbox_h = max(1e-6, y1 - y0)
    cx = (x0 + x1) / 2
    cx_img = img_w / 2


    z = (img_w * real_h) / bbox_h
    x = -((cx - cx_img) * z / img_w)

    return round(x, 2), round(z, 2)


def process_file(path):

    with open(path, "r") as f:
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

        height = float(parts[7])

        latDist, longDist = estimate(x0, y0, x1, y1, height)

        parts[-2] = str(longDist)
        parts[-1] = str(latDist)

        new_lines.append(" ".join(parts))

    # overwrite file
    with open(path, "w") as f:
        f.write("\n".join(new_lines))



def run_batch():

    for file_name in os.listdir(LABELS_FOLDER):
        if not file_name.endswith(".txt"):
            continue

        path = os.path.join(LABELS_FOLDER, file_name)

        process_file(path)

        print(f"Updated: {file_name}")


if __name__ == "__main__":
    run_batch()