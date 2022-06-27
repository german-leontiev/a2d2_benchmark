import torch, pickle, json
from global_names import A2D2_Dataset, sensor_p, abs_, \
get_validation_augmentation, get_preprocessing, A2D2_PATH, preprocessing_fn
import numpy as np
from os.path import join as join_path
from torch.utils.data import DataLoader

with open("bm_ds.pkl", "rb") as f:
    bm_ds = pickle.load(f)

# Load files
with open(join_path(A2D2_PATH, "camera_lidar_semantic", "class_list.json"), "rb") as f:
     class_list= json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ids = bm_ds["test_ids"]
x_test_dir = np.array([abs_(sensor_p(p, "camera")) for p in test_ids])
y_test_dir = np.array([abs_(sensor_p(p, "label")) for p in test_ids])

class_names = list(class_list.values())
class_rgb_values = [[int(i[1:3], 16), int(i[3:5], 16), int(i[5:7], 16)] for i in class_list.keys()]

# Useful to shortlist specific classes in datasets with large number of classes
select_classes = class_names # all classes

# Get RGB values of required classes
select_class_indices = [class_names.index(cls) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

test_dataset = A2D2_Dataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)

model = torch.load('./best_mobilenet_v2_model.pth', map_location=DEVICE)

image, gt_mask = test_dataset[0]
x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

for i in range(10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    pred_mask = model(x_tensor)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(f"Time of prediction: {start.elapsed_time(end):.2f} ms")


import segmentation_models_pytorch as smp
# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

test_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

valid_logs = test_epoch.run(test_dataloader)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")