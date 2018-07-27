from collections import namedtuple
import time
import numpy as np

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
    ["class_weight_path","one_hot_dict_path", "base_path", "train_data_dir", "checkpoint_dir", "train_data_path"]
)
Image = namedtuple(
    "Image",
    ["size", "depth", "mean_subtraction"]
)
Cell = namedtuple(
    "Cell",
    ["num_cells", "cell_size"]
)


model = Model(
    name="olin",
    architecture="CPDr-CPDr-CPDr-DDDDr-L" #Oto for one-by-one
)
hyperparameters = Hyperparameters(
    learning_rate=1e-4,
    batch_size=128,
    num_epochs=5000,
    eval_ratio=0.05,
)
mdyHM_str = time.strftime("%m%d%y%H%M") # e.g. 0706181410 (mmddyyHHMM)
checkpoint_dir_name = "{}_{}-{}_lr{}-bs{}-weighted".format(
    mdyHM_str, model.name, model.architecture, hyperparameters.learning_rate, hyperparameters.batch_size
)
paths = Paths(
    base_path="/home/jlim2/olri_classifier",
    train_data_dir = "frames/moreframes",
    checkpoint_dir = checkpoint_dir_name + "/",
    class_weight_path = "/home/jlim2/olri_classifier/0725181357train_cellcounts-gray-re1.0-en250-max300-submean.npy",
    train_data_path = "/home/jlim2/olri_classifier/0725181357train_data-gray-re1.0-en250-max300-submean.npy",
    one_hot_dict_path = "/home/jlim2/olri_classifier/0725181357train_onehotdict-gray-re1.0-en250-max300-submean.npy"
)

image = Image(
    size=100,
    depth=1,
    mean_subtraction=True
)

np.load(paths.one_hot_dict_path).item().keys()
num_cells = len(np.load(paths.one_hot_dict_path).item().keys())

cell=Cell(cell_size=2,num_cells=num_cells)

"""### Reference
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
























