
"""
Deep Cardiac Phenotyping
3D U-Net Segmentation of Cardiac MRI

Dataset: ACDC cardiac MRI (preprocessed HDF5 volumes)
Frameworks: PyTorch, MONAI

Workflow:
1. Load and preprocess 3D MRI volumes
2. Train a four-class 3D U-Net
3. Visualise segmentation predictions
4. Extract exploratory anatomical measurements
5. Evaluate robustness under simulated Gaussian noise

Research code. Not intended for clinical diagnosis.
"""

from pathlib import Path
import os
import random
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from monai.data import CacheDataset, DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    ToTensord,
)
from monai.utils import set_determinism


# ============================================================
# 1. Configuration and reproducibility
# ============================================================

SEED = 42

set_determinism(seed=SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(
    os.getenv("ACDC_H5_DIR", "./data/ACDC_training_volumes")
)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = int(os.getenv("EPOCHS", "150"))
NOISE_STD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# 2. Custom HDF5 cardiac MRI loader
# ============================================================

class LoadACDCH5d(MapTransform):
    """Load image and segmentation label from an HDF5 file."""

    def __call__(self, data):
        d = dict(data)
        filepath = d[self.keys[0]]

        with h5py.File(filepath, "r") as f:
            image = np.asarray(f["image"][:], dtype=np.float32)
            label = np.asarray(f["label"][:], dtype=np.int64)

        if image.ndim != 3 or label.shape != image.shape:
            raise ValueError(
                f"Unexpected image/label shapes: "
                f"{image.shape} / {label.shape}"
            )

        d["image"] = image[None, ...]
        d["label"] = label[None, ...]

        return d


# ============================================================
# 3. Dataset preparation and preprocessing
# ============================================================

files = sorted(DATA_DIR.glob("*.h5"))

if not files:
    raise FileNotFoundError(
        f"No preprocessed HDF5 volumes found in {DATA_DIR}"
    )

records = [{"h5_path": str(f)} for f in files]

# Original 80/20 file-level split.
# Patient-level separation should be verified separately.

train_size = int(0.8 * len(records))

train_records = records[:train_size]
val_records = records[train_size:]

print("Training volumes:", len(train_records))
print("Validation volumes:", len(val_records))


train_transforms = Compose([
    LoadACDCH5d(keys=["h5_path"]),

    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True,
    ),

    SpatialPadd(
        keys=["image", "label"],
        spatial_size=(16, 256, 240),
    ),

    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(16, 128, 128),
        pos=1,
        neg=1,
        num_samples=4,
    ),

    ToTensord(keys=["image", "label"]),
])


val_transforms = Compose([
    LoadACDCH5d(keys=["h5_path"]),

    NormalizeIntensityd(
        keys=["image"],
        nonzero=True,
        channel_wise=True,
    ),

    SpatialPadd(
        keys=["image", "label"],
        spatial_size=(16, 256, 240),
    ),

    CenterSpatialCropd(
        keys=["image", "label"],
        roi_size=(16, 128, 128),
    ),

    ToTensord(keys=["image", "label"]),
])


train_ds = CacheDataset(
    train_records,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=0,
)

val_ds = CacheDataset(
    val_records,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=0,
)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 4. Four-class 3D U-Net architecture
# ============================================================

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(DEVICE)

loss_fn = DiceCELoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5,
)


# ============================================================
# 5. Model training
# ============================================================

loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, len(train_loader))
    loss_history.append(avg_loss)

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS}: "
        f"loss={avg_loss:.4f}"
    )


checkpoint = OUTPUT_DIR / "unet_acdc_state_dict.pth"

torch.save(model.state_dict(), checkpoint)

pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_loss": loss_history,
}).to_csv(
    OUTPUT_DIR / "training_loss.csv",
    index=False,
)

print("Checkpoint saved:", checkpoint)
print(
    "Training time (minutes):",
    round((time.time() - start_time) / 60, 2),
)


# ============================================================
# 6. Visualise cardiac MRI segmentation
# ============================================================

model.eval()

batch = next(iter(val_loader))

images = batch["image"].to(DEVICE)
labels = batch["label"].to(DEVICE)

with torch.no_grad():
    predicted_masks = model(images).argmax(
        dim=1,
        keepdim=True,
    )

slice_idx = min(8, images.shape[2] - 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

image_slice = images[
    0, 0, slice_idx
].cpu().numpy()

axes[0].imshow(image_slice, cmap="gray")
axes[0].set_title("Cardiac MRI")

for ax, mask, title in [
    (
        axes[1],
        labels[0, 0, slice_idx].cpu().numpy(),
        "Ground-truth mask",
    ),
    (
        axes[2],
        predicted_masks[0, 0, slice_idx].cpu().numpy(),
        "Predicted mask",
    ),
]:

    ax.imshow(image_slice, cmap="gray")

    ax.imshow(
        np.ma.masked_equal(mask, 0),
        cmap="jet",
        alpha=0.5,
        vmin=0,
        vmax=3,
    )

    ax.set_title(title)

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "segmentation_example.png",
    dpi=150,
)

plt.show()


# ============================================================
# 7. Exploratory cardiac morphology measurements
# ============================================================

# These measurements are voxel counts, not physical
# volumes in millilitres. This assumes the label mapping
# used in the original notebook: myocardium = 2,
# left ventricular cavity = 3.

rows = []

model.eval()

with torch.no_grad():

    for index, batch in enumerate(val_loader):

        images = batch["image"].to(DEVICE)

        predicted_masks = model(images).argmax(
            dim=1
        ).cpu().numpy()

        myocardium_voxels = int(
            (predicted_masks == 2).sum()
        )

        lv_cavity_voxels = int(
            (predicted_masks == 3).sum()
        )

        ratio = (
            myocardium_voxels / lv_cavity_voxels
            if lv_cavity_voxels
            else np.nan
        )

        rows.append({
            "validation_volume_index": index,
            "myocardium_voxel_count": myocardium_voxels,
            "lv_cavity_voxel_count": lv_cavity_voxels,
            "exploratory_myocardium_to_lv_ratio": ratio,
        })


measurements = pd.DataFrame(rows)

measurements.to_csv(
    OUTPUT_DIR / "exploratory_morphology_metrics.csv",
    index=False,
)

print(measurements.head())


# ============================================================
# 8. Convert segmentation labels to one-hot masks
# ============================================================

def to_one_hot(mask, num_classes=4):
    """
    Convert (B, 1, D, H, W) integer segmentation masks
    into (B, C, D, H, W) one-hot representations.
    """

    return torch.nn.functional.one_hot(
        mask.squeeze(1).long(),
        num_classes=num_classes,
    ).permute(0, 4, 1, 2, 3).float()


# ============================================================
# 9. Gaussian noise robustness evaluation
# ============================================================

clean_metric = DiceMetric(
    include_background=False,
    reduction="mean",
)

noisy_metric = DiceMetric(
    include_background=False,
    reduction="mean",
)

model.eval()

with torch.no_grad():

    for batch in val_loader:

        images = batch["image"].to(DEVICE)

        ground_truth = to_one_hot(
            batch["label"].to(DEVICE)
        )

        clean_predictions = to_one_hot(
            model(images).argmax(dim=1, keepdim=True)
        )

        noisy_images = (
            images
            + torch.randn_like(images) * NOISE_STD
        )

        noisy_predictions = to_one_hot(
            model(noisy_images).argmax(
                dim=1,
                keepdim=True,
            )
        )

        clean_metric(
            y_pred=clean_predictions,
            y=ground_truth,
        )

        noisy_metric(
            y_pred=noisy_predictions,
            y=ground_truth,
        )


clean_dice = float(clean_metric.aggregate().item())
noisy_dice = float(noisy_metric.aggregate().item())

relative_drop = (
    100 * (clean_dice - noisy_dice) / clean_dice
    if clean_dice
    else float("nan")
)

print(f"Clean Dice: {clean_dice:.3f}")
print(f"Noisy Dice: {noisy_dice:.3f}")
print(f"Relative decrease: {relative_drop:.1f}%")


# ============================================================
# 10. Save robustness evaluation results
# ============================================================

results = pd.DataFrame([{
    "noise_std": NOISE_STD,
    "clean_dice": clean_dice,
    "noisy_dice": noisy_dice,
    "relative_drop_percent": relative_drop,
}])

results.to_csv(
    OUTPUT_DIR / "robustness_metrics.csv",
    index=False,
)

print("Evaluation results saved to:", OUTPUT_DIR)
