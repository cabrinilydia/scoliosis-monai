from dataclasses import dataclass
from pathlib import Path

import monai
import numpy as np
import pydicom
import torch
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
    Transposed,
)
from monai.visualize import plot_2d_or_3d_image
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    image_dir: str = "data/images"
    mask_dir: str = "data/masks_id"
    num_classes: int = 20
    epochs: int = 1000
    batch_size: int = 2
    num_workers: int = 8
    lr: float = 1e-4
    val_ratio: float = 0.2
    seed: int = 42
    val_interval: int = 2
    crop_size: int = 1024
    num_samples_per_image: int = 4
    model_out: str = "best_metric_model_segmentation2d_dict.pth"
    log_dir: str = "runs/unet_training_dict"
    split_by_image: bool = False
    patience: int = 50          # stop if no improvement for this many val checks
    resume: bool = True         # resume from checkpoint if it exists



class LoadDicomd(MapTransform):
    """Load DICOM pixel data into numpy arrays for MONAI dict pipelines."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            ds = pydicom.dcmread(d[key])
            img = ds.pixel_array.astype(np.float32)
            d[key] = img
        return d


def patient_from_stem(stem: str) -> str:
    # Stem format is expected like pt14_I2197000.
    return stem.split("_", 1)[0]


def make_training_dicts(
    image_dir: Path,
    mask_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_by_patient: bool = True,
):
    image_map = {p.stem: p for p in image_dir.glob("*.dcm")}
    mask_map = {p.stem: p for p in mask_dir.glob("*.png")}
    paired = sorted(image_map.keys() & mask_map.keys())

    if not paired:
        raise RuntimeError(f"No matched pairs found in {image_dir} and {mask_dir}")

    samples = []
    for stem in paired:
        samples.append(
            {
                "image": str(image_map[stem]),
                "label": str(mask_map[stem]),
                "patient_id": patient_from_stem(stem),
            }
        )

    if split_by_patient:
        patient_ids = sorted({s["patient_id"] for s in samples})
        train_patients, val_patients = train_test_split(
            patient_ids,
            test_size=val_ratio,
            random_state=seed,
        )
        train_patients = set(train_patients)
        val_patients = set(val_patients)
        train_files = [s for s in samples if s["patient_id"] in train_patients]
        val_files = [s for s in samples if s["patient_id"] in val_patients]
    else:
        train_files, val_files = train_test_split(samples, test_size=val_ratio, random_state=seed)

    return train_files, val_files


def get_transforms(crop_size: int, num_samples: int):
    train_transforms = Compose(
        [
            LoadDicomd(keys=["image"]),
            LoadImaged(keys=["label"]),
            Transposed(keys=["label"], indices=[1, 0]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(crop_size, crop_size),
                pos=1,
                neg=1,
                num_samples=num_samples,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadDicomd(keys=["image"]),
            LoadImaged(keys=["label"]),
            Transposed(keys=["label"], indices=[1, 0]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            CastToTyped(keys=["image"], dtype=np.float32),
            CastToTyped(keys=["label"], dtype=np.int64),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def train(cfg: TrainConfig):
    monai.config.print_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=cfg.log_dir)

    train_files, val_files = make_training_dicts(
        image_dir=Path(cfg.image_dir),
        mask_dir=Path(cfg.mask_dir),
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        split_by_patient=not cfg.split_by_image,
    )
    print(f"Train pairs: {len(train_files)} | Val pairs: {len(val_files)}")
    print("Sample training entry:", {k: v for k, v in train_files[0].items() if k != "patient_id"})

    train_transforms, val_transforms = get_transforms(
        crop_size=cfg.crop_size,
        num_samples=cfg.num_samples_per_image,
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=cfg.num_workers,
        collate_fn=list_data_collate,
    )

    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=cfg.num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.num_classes)
    post_label = AsDiscrete(to_onehot=cfg.num_classes)

    best_metric = -1.0
    best_metric_epoch = -1
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{cfg.epochs} - train_loss: {avg_train_loss:.4f}")
        writer.add_scalar("train_loss", avg_train_loss, epoch + 1)

        if (epoch + 1) % cfg.val_interval != 0:
            continue

        model.eval()
        last_val_images = None
        last_val_labels = None
        last_val_outputs = None
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_outputs = sliding_window_inference(
                    val_images,
                    roi_size=(cfg.crop_size, cfg.crop_size),
                    sw_batch_size=1,
                    predictor=model,
                )
                last_val_images = val_images
                last_val_labels = val_labels
                last_val_outputs = val_outputs
                val_outputs = [post_pred(x) for x in decollate_batch(val_outputs)]
                val_labels_list = [post_label(x) for x in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels_list)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Validation Dice: {metric:.4f}")
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            if last_val_images is not None and last_val_labels is not None and last_val_outputs is not None:
                val_pred_labels = torch.argmax(last_val_outputs, dim=1, keepdim=True)
                plot_2d_or_3d_image(last_val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(last_val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_pred_labels, epoch + 1, writer, index=0, tag="output")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), cfg.model_out)
                print(f"Saved best model to {cfg.model_out}")

    print(f"Training complete. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
    
