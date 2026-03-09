import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
from PIL import Image

import monai
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    ScaleIntensityd,
    Transposed,
)
from sklearn.model_selection import train_test_split


# ── config ────────────────────────────────────────────────────────────
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks_id"
NUM_CLASSES = 20
CROP_SIZE = 1280
VAL_RATIO = 0.2
SEED = 42
SPLIT_BY_IMAGE = False
MODEL_PATH = "best_metric_model_segmentation2d_dict.pth"
OUTPUT_DIR = "eval_outputs"


# ── DICOM loader (same as training) ──────────────────────────────────
class LoadDicomd(MapTransform):
    """Load DICOM pixel data into numpy arrays for MONAI dict pipelines."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            ds = pydicom.dcmread(d[key])
            img = ds.pixel_array.astype(np.float32)
            d[key] = img
        return d


# ── patient splitting (same as training) ─────────────────────────────
def patient_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def make_val_dicts(image_dir, mask_dir, val_ratio=0.2, seed=42, split_by_patient=True):
    image_map = {p.stem: p for p in Path(image_dir).glob("*.dcm")}
    mask_map = {p.stem: p for p in Path(mask_dir).glob("*.png")}
    paired = sorted(image_map.keys() & mask_map.keys())

    samples = [{"image": str(image_map[s]), "label": str(mask_map[s]),
                "patient_id": patient_from_stem(s)} for s in paired]

    if split_by_patient:
        patient_ids = sorted({s["patient_id"] for s in samples})
        _, val_patients = train_test_split(patient_ids, test_size=val_ratio, random_state=seed)
        val_files = [s for s in samples if s["patient_id"] in set(val_patients)]
    else:
        _, val_files = train_test_split(samples, test_size=val_ratio, random_state=seed)

    return val_files


# ── transforms (same preprocessing as training eval/val) ──────────────
val_transforms = Compose([
    LoadDicomd(keys=["image"]),
    LoadImaged(keys=["label"]),
    Transposed(keys=["label"], indices=[1, 0]),        # fix transpose
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    ScaleIntensityd(keys=["image"]),
    CastToTyped(keys=["image"], dtype=np.float32),
    CastToTyped(keys=["label"], dtype=np.int64),
    EnsureTyped(keys=["image", "label"]),
])


# ── overlay helper: background transparent ───────────────────────────
def make_overlay(label_map, num_classes):
    cmap = plt.get_cmap("tab20", num_classes)
    rgba = cmap(label_map / max(num_classes - 1, 1))
    rgba[label_map == 0, 3] = 0  # make background fully transparent
    return rgba


# ── main ──────────────────────────────────────────────────────────────
def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load val set using same split logic as training
    val_files = make_val_dicts(IMAGE_DIR, MASK_DIR, VAL_RATIO, SEED, not SPLIT_BY_IMAGE)
    print(f"Evaluating on {len(val_files)} validation images")

    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=2,
        collate_fn=list_data_collate,
    )

    # load model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            stem = Path(val_files[i]["image"]).stem

            # sliding window inference
            val_outputs = sliding_window_inference(
                val_images,
                roi_size=(CROP_SIZE, CROP_SIZE),
                sw_batch_size=1,
                predictor=model,
            )

            # compute dice
            preds = [post_pred(x) for x in decollate_batch(val_outputs)]
            labels = [post_label(x) for x in decollate_batch(val_labels)]
            dice_metric(y_pred=preds, y=labels)

            # get arrays for plotting
            pred_labels = torch.argmax(val_outputs[0], dim=0).cpu().numpy()  # (H, W)
            gt_labels = val_labels[0, 0].cpu().numpy()                        # (H, W)
            image = val_images[0, 0].cpu().numpy()                            # (H, W)

            # side by side plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 10))

            axes[0].imshow(image, cmap="gray", aspect="auto")
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(image, cmap="gray", aspect="auto")
            axes[1].imshow(make_overlay(gt_labels, NUM_CLASSES), aspect="auto")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(image, cmap="gray", aspect="auto")
            axes[2].imshow(make_overlay(pred_labels, NUM_CLASSES), aspect="auto")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.suptitle(stem, fontsize=12)
            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_eval.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_path}")

    # final dice score
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"\nMean Dice across validation set: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
