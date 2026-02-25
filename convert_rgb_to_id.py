import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

MASK_DIR = "data/masks"
OUT_DIR = "data/masks_id"
os.makedirs(OUT_DIR, exist_ok=True)

# CVAT mapping you pasted (name -> RGB)
LABEL2RGB = {
    "background": (0, 0, 0),
    "C7": (24, 109, 103),
    "T1": (242, 111, 84),
    "T2": (20, 200, 122),
    "T3": (55, 87, 5),
    "T4": (247, 119, 61),
    "T5": (129, 170, 223),
    "T6": (90, 67, 154),
    "T7": (201, 235, 245),
    "T8": (255, 53, 94),
    "T9": (245, 147, 49),
    "T10": (71, 190, 171),
    "T11": (225, 223, 153),
    "T12": (55, 57, 3),
    "L1": (205, 132, 171),
    "L2": (46, 134, 106),
    "L3": (88, 103, 65),
    "L4": (68, 243, 251),
    "L5": (116, 42, 142),
    "S1": (92, 94, 186),
}

# Decide class IDs (stable + human-readable order)
ORDER = [
    "background",
    "C7",
    "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
    "L1","L2","L3","L4","L5",
    "S1",
]

ID2LABEL = {i: name for i, name in enumerate(ORDER)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}
COLOR2ID = {LABEL2RGB[name]: LABEL2ID[name] for name in ORDER}

def convert_one(mask_path: str) -> np.ndarray:
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[:, :, :3]
    else:
        raise ValueError(f"{mask_path} is not RGB. shape={m.shape}")

    h, w, _ = m.shape
    out = np.zeros((h, w), dtype=np.uint8)
    mapped = np.zeros((h, w), dtype=bool)

    for (r, g, b), cid in COLOR2ID.items():
        match = (m[..., 0] == r) & (m[..., 1] == g) & (m[..., 2] == b)
        out[match] = cid
        mapped |= match

    if not mapped.all():
        unknown = np.unique(m[~mapped].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown colors found (not in CVAT mapping): {unknown[:20]}")

    return out

def main():
    files = sorted([f for f in os.listdir(MASK_DIR) if f.lower().endswith(".png")])
    for f in tqdm(files, desc="Converting masks"):
        src = os.path.join(MASK_DIR, f)
        dst = os.path.join(OUT_DIR, f)
        out = convert_one(src)
        Image.fromarray(out).save(dst)

    # save mapping files for training/visualization later
    with open(os.path.join(OUT_DIR, "id2label.json"), "w") as fp:
        json.dump({str(k): v for k, v in ID2LABEL.items()}, fp, indent=2)

    with open(os.path.join(OUT_DIR, "label2rgb.json"), "w") as fp:
        json.dump({k: list(v) for k, v in LABEL2RGB.items()}, fp, indent=2)

    print("✅ Done.")
    print("ID masks ->", OUT_DIR)
    print("Saved:", os.path.join(OUT_DIR, "id2label.json"))
    print("Total classes:", len(ORDER), "(0..19)")

if __name__ == "__main__":
    main()