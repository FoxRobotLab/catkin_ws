"""
See YOLO docs for fine-tuning: https://docs.ultralytics.com/guides/finetuning-guide/#choosing-a-model-size

Using optimizer=AdamW and lr=0.001 is recommended for small datasets, and freeze=10 is recommended, when dataset is
small and of a similar domain to COCO.

Using device="mps" is suggested here: https://docs.ultralytics.com/modes/train/#apple-silicon-mps-training

Training is actually way faster using Google's T4 GPU on Colab. This code can be adapted for a Colab notebook.
"""

from ultralytics import YOLO
from datetime import datetime

TASK_NAME = "all"
TRAIN_EPOCHS = 20
TRAIN_OPTIMIZER = "AdamW"
TRAIN_LR = 0.001
TRAIN_FRZ = 10
OUTPUT_NAME = f"{datetime.now().strftime('%m%d%Y')}_{TASK_NAME}_Ep{TRAIN_EPOCHS}_{TRAIN_OPTIMIZER}_LR{TRAIN_LR}_Frz{TRAIN_FRZ}"

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model

# Train the model, using mps for M1 Mac
results = model.train(data="/Users/oscarrezab/GitHub/macalester/catkin_ws/src/collision_avoidance/res/train_data_annotated_all/data.yaml",
                      epochs=TRAIN_EPOCHS, imgsz=640, optimizer=TRAIN_OPTIMIZER, lr0=TRAIN_LR, freeze=TRAIN_FRZ, patience=5, device="mps",
                      project="/Users/oscarrezab/GitHub/macalester/catkin_ws/src/collision_avoidance/scripts/training/trained_models/",
                      name=OUTPUT_NAME, conf=0.25, agnostic_nms=True)

model.export(format="onnx")
