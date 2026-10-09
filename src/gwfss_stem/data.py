"""GWFSS data preparation, splits, and PyTorch dataset."""
import glob
import io
import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

HF_REPO = "GlobalWheat/GWFSS_v1.0"
CLASSES = ["background", "head", "stem", "leaf"]
STEM = 2
IGNORE = 255
# Verified against GlobalWheat/GWFSS_model_v1.0/class.json
COLOR_TO_ID = {(0, 0, 0): 0, (50, 255, 132): 1, (50, 132, 255): 2, (214, 255, 50): 3}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Region split exactly as in the GWFSS paper (Plant Phenomics 2025, Sec. 2.5.1).
# Verified against the PDF: USASK is not used in the paper's region split.
REGION_SPLIT = {
    "train": ["Arvalis", "CIMMYT", "ETHZ", "INRAE", "NJAU", "RRES", "ULiege"],
    "val": ["UTokyo"],
    "test": ["UQ"],
}


def institute_of(name):
    return name.split("_")[0]


def rgb_to_index(rgb):
    key = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    out = np.full(rgb.shape[:2], IGNORE, dtype=np.uint8)
    for (r, g, b), cid in COLOR_TO_ID.items():
        out[key == ((r << 16) | (g << 8) | b)] = cid
    return out


def prepare_dataset(data_root, parquet_dir=None):
    """Decode the labelled split into data_root/{images,masks} and write manifest.csv.

    Masks are stored as single-channel class-index PNGs (0..3, 255 = unknown colour).
    If parquet_dir is None the parquet shards are downloaded from the Hugging Face Hub.
    """
    import pyarrow.parquet as pq

    manifest_path = os.path.join(data_root, "manifest.csv")
    if os.path.exists(manifest_path):
        return pd.read_csv(manifest_path)
    if parquet_dir is None:
        from huggingface_hub import snapshot_download

        parquet_dir = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=["data/labelled-*"])
        parquet_dir = os.path.join(parquet_dir, "data")
    files = sorted(glob.glob(os.path.join(parquet_dir, "labelled-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    os.makedirs(os.path.join(data_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "masks"), exist_ok=True)
    rows = []
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g).to_pylist():
                name = row["mask"]["path"]
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                rgb = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])).convert("RGB"))
                idx = rgb_to_index(rgb)
                img.save(os.path.join(data_root, "images", name))
                Image.fromarray(idx).save(os.path.join(data_root, "masks", name))
                counts = np.bincount(idx.ravel(), minlength=256)
                rows.append(dict(
                    name=name, institute=institute_of(name), height=idx.shape[0], width=idx.shape[1],
                    bg_px=int(counts[0]), head_px=int(counts[1]), stem_px=int(counts[2]),
                    leaf_px=int(counts[3]), unknown_px=int(counts[IGNORE]),
                    empty_mask=bool(counts[0] == idx.size),
                ))
    df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    df.to_csv(manifest_path, index=False)
    return df


def make_splits(manifest, seed=42, ratios=(0.7, 0.1, 0.2)):
    """Return dict of splits: random split stratified by institute, region split, and LOIO folds."""
    rng = random.Random(seed)
    rand = {"train": [], "val": [], "test": []}
    for inst, grp in manifest.groupby("institute"):
        names = sorted(grp["name"].tolist())
        rng.shuffle(names)
        n = len(names)
        n_tr, n_va = round(ratios[0] * n), round(ratios[1] * n)
        rand["train"] += names[:n_tr]
        rand["val"] += names[n_tr:n_tr + n_va]
        rand["test"] += names[n_tr + n_va:]
    splits = {"random": {k: sorted(v) for k, v in rand.items()}}
    by_inst = manifest.groupby("institute")["name"].apply(lambda s: sorted(s)).to_dict()
    splits["region"] = {k: sorted(n for i in insts for n in by_inst.get(i, [])) for k, insts in REGION_SPLIT.items()}
    insts = sorted(by_inst)
    for i, held in enumerate(insts):
        val_inst = insts[(i + 1) % len(insts)]
        splits[f"loio_{held}"] = {
            "train": sorted(n for j in insts if j not in (held, val_inst) for n in by_inst[j]),
            "val": by_inst[val_inst],
            "test": by_inst[held],
        }
    return splits


def save_splits(splits, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)


def load_split(path, name):
    with open(path) as f:
        return json.load(f)[name]


def stem_skeleton(mask, dilate=True):
    """Tubed skeleton of the stem class (Skeleton Recall Loss target)."""
    sk = skeletonize(mask == STEM).astype(np.uint8)
    if dilate:
        sk = cv2.dilate(sk, np.ones((3, 3), np.uint8))
    sk[mask == IGNORE] = 0
    return sk


class SegDataset(Dataset):
    def __init__(self, data_root, names, train=False, crop=512, scale=1.0, with_skeleton=False):
        self.root, self.names = data_root, list(names)
        self.train, self.crop, self.scale, self.with_skeleton = train, crop, scale, with_skeleton

    def __len__(self):
        return len(self.names)

    def load(self, name):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, "masks", name), cv2.IMREAD_UNCHANGED)
        if self.scale != 1.0:
            h, w = mask.shape
            size = (round(w * self.scale), round(h * self.scale))
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def augment(self, img, mask):
        c = self.crop
        h, w = mask.shape
        ph, pw = max(0, c - h), max(0, c - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=IGNORE)
            h, w = mask.shape
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1], mask[::-1]
        k = random.randint(0, 3)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
        img = img.astype(np.float32)
        if random.random() < 0.5:
            alpha, beta = random.uniform(0.8, 1.2), random.uniform(-20, 20)
            img = np.clip(img * alpha + beta, 0, 255)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        name = self.names[i]
        img, mask = self.load(name)
        if self.train:
            img, mask = self.augment(img, mask)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        out = {
            "image": torch.from_numpy(img.transpose(2, 0, 1).copy()),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "name": name,
        }
        if self.with_skeleton:
            out["skeleton"] = torch.from_numpy(stem_skeleton(mask).astype(np.float32))
        return out
