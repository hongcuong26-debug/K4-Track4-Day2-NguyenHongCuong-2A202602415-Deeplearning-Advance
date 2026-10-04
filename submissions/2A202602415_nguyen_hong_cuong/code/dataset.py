"""DeepWeeds data loading, split validation, transforms, and DataLoaders."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 9
EXPECTED_IMAGES = 17_509
# The author's fold CSVs contain Filename/Label; labels.csv additionally has Species.
REQUIRED_COLUMNS = {"Filename", "Label"}
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_frame(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if df["Filename"].duplicated().any():
        duplicates = df.loc[df["Filename"].duplicated(), "Filename"].head().tolist()
        raise ValueError(f"{name} contains duplicate filenames: {duplicates}")
    labels = pd.to_numeric(df["Label"], errors="raise")
    if not labels.between(0, NUM_CLASSES - 1).all():
        raise ValueError(f"{name} labels must be in [0, {NUM_CLASSES - 1}]")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Load the author-provided train/validation/test CSVs without re-splitting."""
    if fold not in range(5):
        raise ValueError("fold must be one of 0, 1, 2, 3, 4")
    root = Path(labels_dir)
    frames = []
    for split in ("train", "val", "test"):
        path = root / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        _validate_frame(frame, path.name)
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Fail fast on leakage, incomplete folds, bad labels, and missing images."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in frames.items():
        _validate_frame(frame, name)
    names = {key: set(df["Filename"].astype(str)) for key, df in frames.items()}
    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    leaking = {key: value for key, value in overlap.items() if value}
    if leaking:
        raise ValueError(f"split leakage detected: { {k: v[:5] for k, v in leaking.items()} }")
    union = set().union(*names.values())
    if len(union) != EXPECTED_IMAGES:
        raise ValueError(f"fold must cover {EXPECTED_IMAGES} unique images, found {len(union)}")
    image_root = Path(images_dir)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    missing_files = sorted(name for name in union if not (image_root / name).is_file())
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} CSV images are missing from {image_root}; first: {missing_files[:10]}"
        )
    result = {
        "n": {key: len(df) for key, df in frames.items()},
        "per_class": {
            key: {int(i): int(df["Label"].eq(i).sum()) for i in range(NUM_CLASSES)}
            for key, df in frames.items()
        },
        "overlap": {key: len(value) for key, value in overlap.items()},
        "union": len(union),
        "missing_files": 0,
    }
    print(pd.DataFrame(result["per_class"]).rename_axis("Label"))
    print("split sizes:", result["n"], "overlap:", result["overlap"], "union:", result["union"])
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic evaluation transforms or one of four train policies."""
    from torchvision import transforms as T
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(img_size), T.ToTensor(),
                          T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    policies = {
        "basic": [],
        "color": [T.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.05)],
        "trivial": [T.TrivialAugmentWide()],
        "randaug": [T.RandAugment(num_ops=2, magnitude=9)],
    }
    if aug not in policies:
        raise ValueError(f"unknown augmentation {aug!r}; choose from {sorted(policies)}")
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(), *policies[aug], T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepWeedsDataset(Dataset):
    """Map CSV rows to ``(image_tensor, integer_label, filename)``."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        _validate_frame(df, "dataset")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Create an ordered evaluation loader or reproducible shuffled/balanced train loader."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    sample_strategy = None
    shuffle = bool(train)
    if sampler is not None:
        if sampler != "balanced":
            raise ValueError("sampler must be None or 'balanced'")
        if not train:
            raise ValueError("balanced sampling is only valid for training")
        labels = df["Label"].to_numpy(dtype=int)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        if np.any(counts == 0):
            raise ValueError("balanced sampler requires every class in the training split")
        weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        sample_strategy = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sample_strategy,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=bool(train and len(dataset) >= batch_size),
        persistent_workers=num_workers > 0, worker_init_fn=_seed_worker,
    )
