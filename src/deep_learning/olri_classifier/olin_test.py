from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import time
import numpy as np
import rospy
import keras
import olin_factory as factory
from datetime import datetime

recent_n_cells = []
olin_map = cv2.imread(factory.paths.map_path)
cellF = open(factory.paths.maptocell_path, 'r')
cellDict = dict()
for line in cellF:
    if line[0] == '#' or line.isspace():
        continue
    parts = line.split()
    cellNum = parts[0]
    locList = [int(v) for v in parts[1:]]
    cellDict[cellNum] = locList
cell_data =  cellDict

# The following reads in the cell data, in case we want to display it
def _convertWorldToPixels((worldX, worldY)):
    """Converts coordinates in meters in the world to integer coordinates on the map
    Note that this also has to adjust for the rotation and flipping of the map."""
    # First convert from meters to pixels, assuming 20 pixels per meter
    pixelX = worldX * 20.0
    pixelY = worldY * 20.0
    # Next flip x and y values around
    mapX = 2142 - 1 - pixelY
    mapY = 815 - 1 - pixelX
    return (int(mapX), int(mapY))


def drawBox(image, lrpt, ulpt, color, thickness=1):
    """Draws a box at a position given by lower right and upper left locations,
    with the given color."""
    mapUL = _convertWorldToPixels(ulpt)
    mapLR = _convertWorldToPixels(lrpt)
    cv2.rectangle(image, mapUL, mapLR, color, thickness=thickness)


def hightlightCell(image, cellNum, color=(113, 179, 60)):
    """Takes in a cell number and draws a box around it to highlight it."""
    [x1, y1, x2, y2] = cell_data[cellNum]
    drawBox(image, (x1, y1), (x2, y2), color, 2)


def test_turtlebot(olin_classifier, recent_n_max):
    cell_to_intlabel_dict = np.load(factory.paths.one_hot_dict_path).item()
    intlabel_to_cell_dict = dict()
    for cell in cell_to_intlabel_dict.keys():
        intlabel_to_cell_dict[cell_to_intlabel_dict[cell]] = cell
    print(intlabel_to_cell_dict)
    mean = np.load(olin_classifier.paths.train_mean_path)
    ### Load model and weights from the specified checkpoint
    model = keras.models.load_model(olin_classifier.checkpoint_name)
    model.load_weights(olin_classifier.checkpoint_name)
    print("*** Model restored: ", olin_classifier.checkpoint_name)
    model.summary()
    softmax = keras.models.Model(inputs=model.input, outputs=model.get_layer(name="dense_3").output)

    while (not rospy.is_shutdown()):
        turtle_image, _ = olin_classifier.robot.getImage()
        # turtle_image = cv2.imread("/home/macalester/PycharmProjects/olri_classifier/frames/lessframes/frame0076.jpg")
        resized_image = cv2.resize(turtle_image, (olin_classifier.image.size, olin_classifier.image.size))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        submean_image = np.subtract(gray_image, mean)
        cleaned_image = np.array([submean_image], dtype="float")\
            .reshape(1, olin_classifier.image.size, olin_classifier.image.size, olin_classifier.image.depth)
        pred = softmax.predict(cleaned_image)
        pred_class = np.argmax(pred)
        pred_cell = intlabel_to_cell_dict[int(pred_class)]
        ### Compute the best prediction by getting the mode out of most recent n cells
        best_cell = mode_from_recent_n(recent_n_max, pred_cell)
        print("{} Predicted Cell: ".format(datetime.now()), pred_cell)
        print("{} Predicted Best Cell (N={}): ".format(datetime.now(), len(recent_n_cells)), best_cell)
        olin_map_copy = olin_map.copy()
        hightlightCell(olin_map_copy, str(pred_cell))
        hightlightCell(olin_map_copy, str(best_cell), color=(254, 127, 156))

        cv2.imshow("Map Image", olin_map_copy)
        # cv2.imshow("Test Image", turtle_image)
        cv2.imshow("Cleaned Image", submean_image)

        key = cv2.waitKey(10)
        ch = chr(key & 0xFF)
        if (ch == "q"):
            break
        time.sleep(0.1)
    cv2.destroyAllWindows()
    olin_classifier.robot.stop()


def mode_from_recent_n(n, recent_cell):
    """
    Add the most recent cell to the queue that holds most recent N cells and find out the mode of it in order to
    get the best prediction (Jane Pellegrini, JJ Lim)
    :param recent_cell: cell to be added to the queue
    :return: mode of the recent N cells
    """
    recent_n_cells.append(recent_cell)
    if (len(recent_n_cells) > n):
        recent_n_cells.pop(0)
    return max(set(recent_n_cells), key=recent_n_cells.count) # O(n**2)

# if __name__ == "__main__":
#     minor_cells = [
#         18, 19, 67, 69, 71, 73, 75, 95, 97, 99,
#         101, 104, 105, 106, 107, 108, 109, 110, 111, 112,
#         113, 114, 115, 116, 117, 119, 121, 123, 125, 126, 127, 128, 133, 137, 142, 144, 145, 146, 147, 148, 149, 150, 151]
#     cv2.imshow("Olin-Rice Map with Cell Prediction Highlighted", olin_map)