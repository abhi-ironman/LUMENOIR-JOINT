"""Paired and zero-reference data loading for Zero-DCE++ fine-tuning.

Training and validation records come from the materialized split manifests.
A record may be paired (LOL) or single-image (SICE Part 1). The model's
zero-reference losses use only the low-light input; ground truth is retained
for paired validation metrics, never for the training objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training.data_pipeline.config import PROJECT_ROOT
from training.data_pipeline.preprocessing import (
    apply_darkening,
    augment_views,
    load_rgb_float32,
    random_crop,
    random_crop_pair,
)


TEST_PATH_PARTS = frozenset({"eval15", "test", "part2_lowlight", "do_no_harm"})


@dataclass(frozen=True)
class PairRecord:
    """One low-light image with an optional paired reference."""

    dataset: str
    image_id: str
    low: Path
    high: Path | None

    @property
    def has_reference(self) -> bool:
        return self.high is not None


def _resolve(raw: str, base: Path, data_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    for candidate in (
        base / path,
        data_root / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return (base / path).resolve()


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"]]
    if isinstance(payload.get("datasets"), list):
        records: list[dict[str, Any]] = []
        for dataset in payload["datasets"]:
            key = dataset.get("dataset") or dataset.get("key") or "validation"
            for item in dataset.get("records", []):
                record = dict(item)
                record.setdefault("dataset", key)
                records.append(record)
        return records
    raise ValueError("Manifest has no 'records' or 'datasets' list.")


def is_test_path(path: Path, data_root: Path) -> bool:
    """Reject official test and do-no-harm folders from train/validation."""
    try:
        parts = path.resolve().relative_to(data_root.resolve()).parts
    except ValueError:
        parts = path.resolve().parts
    return any(part.lower() in TEST_PATH_PARTS for part in parts)


def _pool_matches_expected(pool: str | None, expected_split: str) -> bool:
    if expected_split == "train":
        # The main dataset pipeline names this pool "fine_tune"; the legacy
        # paired-only split builder names it "train".
        return pool in {"train", "fine_tune"}
    return pool == expected_split


def load_pairs(
    manifest_path: Path,
    *,
    expected_split: str,
    data_root: Path,
    allow_unpaired: bool = True,
) -> list[PairRecord]:
    """Load paired LOL and optional single-image SICE records.

    Single-image records are represented by high=None and has_gt=False. They
    are valid for zero-reference train/validation loss, but not reference
    metrics. The caller must select paired records explicitly for PSNR/SSIM/LPIPS.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool = payload.get("pool")
    pairs: list[PairRecord] = []
    seen: set[tuple[str, str]] = set()

    for index, record in enumerate(_records(payload)):
        raw_split = record.get("split")
        if raw_split is not None:
            split_matches = raw_split == expected_split
        else:
            split_matches = _pool_matches_expected(pool, expected_split)
        if not split_matches:
            raise ValueError(
                f"{manifest_path}: record {index} has split/pool "
                f"{raw_split or pool!r}; expected {expected_split!r}."
            )

        kind = record.get("kind")
        if kind not in (None, "paired", "unpaired"):
            raise ValueError(f"{manifest_path}: record {index} has unknown kind {kind!r}.")

        low_raw = record.get("low")
        high_raw = record.get("high")
        has_gt = bool(record.get("has_gt", bool(high_raw)))
        if not low_raw:
            raise ValueError(f"{manifest_path}: record {index} is missing low.")
        if has_gt and not high_raw:
            raise ValueError(
                f"{manifest_path}: record {index} declares has_gt but has no high path."
            )
        if kind == "paired" and not high_raw:
            raise ValueError(f"{manifest_path}: paired record {index} is missing high.")
        if kind == "unpaired" and high_raw:
            raise ValueError(
                f"{manifest_path}: unpaired record {index} unexpectedly has a high path."
            )
        if not high_raw and not allow_unpaired:
            raise ValueError(
                f"{manifest_path}: record {index} is unpaired; a reference is required."
            )

        base = manifest_path.parent
        low = _resolve(str(low_raw), base, data_root)
        high = _resolve(str(high_raw), base, data_root) if high_raw else None
        if not low.is_file() or (high is not None and not high.is_file()):
            raise ValueError(
                f"{manifest_path}: record {index} points to a missing file: "
                f"{low} / {high}"
            )
        if is_test_path(low, data_root) or (
            high is not None and is_test_path(high, data_root)
        ):
            raise ValueError(f"{manifest_path}: test path or do-no-harm path in {expected_split}: {low}")

        if high is not None:
            with Image.open(low) as low_image, Image.open(high) as high_image:
                if low_image.size != high_image.size:
                    raise ValueError(
                        f"{manifest_path}: size mismatch at record {index}: "
                        f"{low_image.size} vs {high_image.size}"
                    )

        dataset = str(record.get("dataset") or "unknown")
        image_id = str(record.get("image_id") or record.get("id") or low.name)
        key = (dataset, image_id)
        if key in seen:
            raise ValueError(f"{manifest_path}: duplicate record {key!r}.")
        seen.add(key)
        pairs.append(PairRecord(dataset, image_id, low, high))

    if not pairs:
        raise ValueError(f"{manifest_path}: no usable records.")
    return sorted(pairs, key=lambda item: (item.dataset, item.image_id))


def _record_paths(record: PairRecord) -> set[Path]:
    paths = {record.low.resolve()}
    if record.high is not None:
        paths.add(record.high.resolve())
    return paths


def load_train_validation_pairs(
    train_manifest: Path,
    validation_manifest: Path,
    data_root: Path,
) -> tuple[list[PairRecord], list[PairRecord]]:
    train = load_pairs(
        train_manifest,
        expected_split="train",
        data_root=data_root,
        allow_unpaired=True,
    )
    validation = load_pairs(
        validation_manifest,
        expected_split="validation",
        data_root=data_root,
        allow_unpaired=True,
    )
    train_paths = set().union(*(_record_paths(pair) for pair in train))
    validation_paths = set().union(*(_record_paths(pair) for pair in validation))
    overlap = train_paths & validation_paths
    if overlap:
        raise ValueError(f"Train/validation path leakage: {len(overlap)} file(s).")
    return train, validation


def _ensure_crop_size(
    low: np.ndarray, high: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = low.shape[:2]
    if min(height, width) >= size:
        return low, high
    scale = size / min(height, width)
    target = (max(size, round(width * scale)), max(size, round(height * scale)))
    return (
        cv2.resize(low, target, interpolation=cv2.INTER_LINEAR),
        cv2.resize(high, target, interpolation=cv2.INTER_LINEAR),
    )


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


class PairedDataset(Dataset):
    """Training input dataset; accepts paired LOL and unpaired SICE records."""

    def __init__(
        self,
        pairs: list[PairRecord],
        *,
        crop_size: int = 384,
        train: bool,
        seed: int = 42,
        augment: bool = True,
        darkening_probability: float = 0.5,
    ) -> None:
        if not 0.0 <= darkening_probability <= 1.0:
            raise ValueError("darkening_probability must be in [0, 1].")
        self.pairs = pairs
        self.crop_size = crop_size
        self.train = train
        self.seed = seed
        self.augment = augment and train
        self.darkening_probability = darkening_probability if train else 0.0
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        low = load_rgb_float32(pair.low)
        high = load_rgb_float32(pair.high) if pair.high is not None else None
        if high is not None and low.shape != high.shape:
            raise ValueError(f"Pair shape mismatch for {pair.image_id}.")

        if self.train:
            rng = np.random.default_rng(self.seed + self.epoch * len(self.pairs) + index)
            if high is not None:
                low, high = _ensure_crop_size(low, high, self.crop_size)
                low, high = random_crop_pair(low, high, self.crop_size, rng)
            else:
                low, _ = _ensure_crop_size(low, low, self.crop_size)
                low = random_crop(low, self.crop_size, rng)
            if self.augment:
                view = int(rng.integers(0, 8))
                low = augment_views(low)[view]
                if high is not None:
                    high = augment_views(high)[view]
            low = apply_darkening(low, rng, probability=self.darkening_probability)

        return {
            "low": _to_tensor(low),
            "high": _to_tensor(high) if high is not None else None,
            "dataset": pair.dataset,
            "image_id": pair.image_id,
        }


def _collate_training_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate zero-reference inputs; paired references are not training targets."""
    return {
        "low": torch.stack([item["low"] for item in items]),
        "dataset": [item["dataset"] for item in items],
        "image_id": [item["image_id"] for item in items],
        "has_reference": [item["high"] is not None for item in items],
    }


def make_loader(
    dataset: PairedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    epoch: int,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=_collate_training_batch,
    )


__all__ = [
    "PairRecord",
    "PairedDataset",
    "is_test_path",
    "load_pairs",
    "load_train_validation_pairs",
    "make_loader",
]
