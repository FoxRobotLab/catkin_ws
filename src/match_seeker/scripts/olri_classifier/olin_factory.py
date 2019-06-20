from collections import namedtuple
import time
import numpy as np

### Make namedtuples
Model = namedtuple(
    "Model",
    ["name", "architecture"]
)
Hyperparameters = namedtuple(
    "Hyperparameters",
    ["learning_rate", "batch_size", "num_epochs", "eval_ratio"]
)
Paths = namedtuple(
    "Paths",
    ["map_path", "maptocell_path", "one_hot_dict_path", "base_path", "train_data_dir", "test_data_dir", "checkpoint_dir",
     "cell_graph_path", "cell_data_path", "train_data_path", "train_mean_path", "class_weight_path", "checkpoint_name"]
)
Image = namedtuple(
    "Image",
    ["size", "depth", "mean_subtraction"]
)
Cell = namedtuple(
    "Cell",
    ["num_cells", "cell_size", "num_max_cells"]
)

### Initialize namedtuples
model = Model(
    name="olin",
    architecture="CPDrCPDrDDDrL"
)
hyperparameters = Hyperparameters(
    learning_rate=1e-3,
    batch_size=100,
    num_epochs=2000,
    eval_ratio=0.1,
)
mdyHM_str = time.strftime("%m%d%y%H%M") # e.g. 0706181410 (mmddyyHHMM)
checkpoint_dir_name = "{}_lr{}-bs{}".format(
    mdyHM_str, hyperparameters.learning_rate, hyperparameters.batch_size
)
paths = Paths(
    base_path="/home/macalester/PycharmProjects/olri_classifier",
    train_data_dir = "frames/moreframes",
    test_data_dir = "",
    checkpoint_dir = checkpoint_dir_name + "/",
    map_path = "OlinGraphCellMap.png",
    cell_graph_path = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/map/cellGraph.txt",
    maptocell_path = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/res/map/mapToCells.txt",
    cell_data_path = '/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/THE_MASTER_CELL_LOC_FRAME_IDENTIFIER.txt',#"/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/frames/morecells.txt",
    train_data_path = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181357train_data-gray-re1.0-en250-max300-submean.npy",
    class_weight_path= "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181357train_cellcounts-gray-re1.0-en250-max300-submean.npy",
    train_mean_path = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181357train_mean-gray-re1.0-en250-max300-submean.npy",
    one_hot_dict_path = "/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181357train_onehotdict-gray-re1.0-en250-max300-submean.npy",
    checkpoint_name="/home/macalester/PycharmProjects/catkin_ws/src/match_seeker/scripts/olri_classifier/0725181447_olin-CPDr-CPDr-CPDr-DDDDr-L_lr0.001-bs128-weighted/00-745-0.71.hdf5"
)


image = Image(
    size=100,
    depth=1,
    mean_subtraction=True
)
cell_graph = open(paths.cell_graph_path, "r")
for line in cell_graph:
    if line == "" or line.isspace() or line[0] == '#':
        continue
    else:
        if line.startswith("Number of Nodes"):
            num_max_cells = int(line.split()[-1])
            break
cell_graph.close()

np.load(paths.one_hot_dict_path).item().keys()
num_cells = len(np.load(paths.one_hot_dict_path).item().keys())
cell = Cell(
    cell_size=2,
    num_cells=num_cells,
    num_max_cells=num_max_cells
)
#print("Num cells:", num_cells,"Num max cells:",num_max_cells)
"""
### Reference
Usage: https://www.geeksforgeeks.org/namedtuple-in-python/
namedtuple vs. dict: https://stackoverflow.com/questions/9872255/when-and-why-should-i-use-a-namedtuple-instead-of-a-dictionary

### To check fields and values
print(hyperparameters)
Hyperparameters(learning_rate=0.0001, batch_size=32, num_epochs=10, eval_ratio=0.1)

### To check all the fields
print("FIELDS", hyperparameters._fields)
('learning_rate', 'batch_size', 'num_epochs', 'eval_ratio')

### To change a value
print(hyperparameters)
print(hyperparameters._replace(batch_size=10)) # changed here
print(hyperparameters) # back to original
Hyperparameters(learning_rate=0.0001, batch_size=32, num_epochs=10, eval_ratio=0.1)
Hyperparameters(learning_rate=0.0001, batch_size=10, num_epochs=10, eval_ratio=0.1)
Hyperparameters(learning_rate=0.0001, batch_size=32, num_epochs=10, eval_ratio=0.1)
"""
























