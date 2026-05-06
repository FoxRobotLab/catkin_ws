import cv2
import numpy as np


#pixel coordinates from image
image_points = np.array([
    [120, 460],  # bottom-left
    [520, 460],  # bottom-right
    [580, 120],  # top-right
    [80, 120]    # top-left
], dtype=np.float32)

# Real-world coordinates using tiles(estimated from images)
world_points = np.array([
    [0, 0],
    [2, 0],
    [2, 8],
    [0, 8]
], dtype=np.float32)


h, _ = cv2.findHomography(image_points, world_points)

np.save("homography.npy", h)

print("Homography saved")