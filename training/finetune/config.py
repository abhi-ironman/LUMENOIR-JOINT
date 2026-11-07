"""Fine-tuning configuration for Zero-DCE++.

The defaults follow ``final_decision.md`` §2.6 and the verified official
Zero-DCE++ training defaults recorded in ``docs/zero_dce.md``. They are data,
not hidden behaviour: every loss weight, schedule and early-stopping value is
visible in the JSON configuration and in ``run_config.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "training" / "finetune" / "configs" / "zero_dce_plus_plus_lol.json"
)


@dataclass(frozen=True)
class LossWeights:
    """Zero-DCE++ losses plus the documented LUMENOIR additions."""

    spatial: float = 1.0
    exposure: float = 10.0
    color: float = 3.5
    tv_alpha: float = 1600.0
    highlight: float = 5.0
    hue: float = 2.0
    exposure_target: float = 0.6


@dataclass(frozen=True)
class SchedulerSettings:
    name: str = "reduce_on_plateau"
    factor: float = 0.5
    patience: int = 2
    min_lr: float = 1e-6


@dataclass(frozen=True)
class EarlyStoppingSettings:
    patience: int = 5
    min_delta: float = 0.002


@dataclass(frozen=True)
class TrainConfig:
    train_manifest: Path
    validation_manifest: Path
    data_root: Path
    pretrained_weights: Path
    output_dir: Path
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    batch_size: int = 8
    num_workers: int = 4
    crop_size: int = 384
    darkening_probability: float = 0.5
    scale_factor: int = 1
    epochs: int = 30
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.1
    metrics_device: str = "cuda"
    validation_metrics: tuple[str, ...] = ("psnr", "ssim", "lpips")
    max_train_images: int | None = None
    max_val_images: int | None = None
    log_every: int = 10
    save_every: int = 1
    resume: Path | None = None
    loss: LossWeights = field(default_factory=LossWeights)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    early_stopping: EarlyStoppingSettings = field(default_factory=EarlyStoppingSettings)

    def validate(self) -> None:
        if not self.train_manifest.is_file():
            raise ValueError(f"Training manifest not found: {self.train_manifest}")
        if not self.validation_manifest.is_file():
            raise ValueError(f"Validation manifest not found: {self.validation_manifest}")
        if not self.pretrained_weights.is_file():
            raise ValueError(f"Pretrained checkpoint not found: {self.pretrained_weights}")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.crop_size < 8:
            raise ValueError("crop_size must be at least 8.")
        if not 0.0 <= self.darkening_probability <= 1.0:
            raise ValueError("darkening_probability must be in [0, 1].")
        if not 1 <= self.scale_factor <= 12:
            raise ValueError("scale_factor must be in [1, 12].")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.max_train_images is not None and self.max_train_images <= 0:
            raise ValueError("max_train_images must be positive when set.")
        if self.max_val_images is not None and self.max_val_images <= 0:
            raise ValueError("max_val_images must be positive when set.")
        if self.pretrained_weights.resolve() == (self.output_dir / "best.pth").resolve():
            raise ValueError("best.pth must not overwrite the pretrained checkpoint.")
        if self.resume is not None and not self.resume.is_file():
            raise ValueError(f"Resume checkpoint not found: {self.resume}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_manifest": str(self.train_manifest),
            "validation_manifest": str(self.validation_manifest),
            "data_root": str(self.data_root),
            "pretrained_weights": str(self.pretrained_weights),
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "device": self.device,
            "amp": self.amp,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "crop_size": self.crop_size,
            "darkening_probability": self.darkening_probability,
            "scale_factor": self.scale_factor,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "metrics_device": self.metrics_device,
            "validation_metrics": list(self.validation_metrics),
            "max_train_images": self.max_train_images,
            "max_val_images": self.max_val_images,
            "log_every": self.log_every,
            "save_every": self.save_every,
            "resume": str(self.resume) if self.resume else None,
            "loss": {
                "spatial": self.loss.spatial,
                "exposure": self.loss.exposure,
                "color": self.loss.color,
                "tv_alpha": self.loss.tv_alpha,
                "highlight": self.loss.highlight,
                "hue": self.loss.hue,
                "exposure_target": self.loss.exposure_target,
            },
            "scheduler": {
                "name": self.scheduler.name,
                "factor": self.scheduler.factor,
                "patience": self.scheduler.patience,
                "min_lr": self.scheduler.min_lr,
            },
            "early_stopping": {
                "patience": self.early_stopping.patience,
                "min_delta": self.early_stopping.min_delta,
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        base: Path = PROJECT_ROOT,
    ) -> TrainConfig:
        def path_value(key: str) -> Path:
            raw = Path(str(payload[key]))
            if raw.is_absolute():
                return raw
            candidate = base / raw
            return candidate if candidate.exists() else PROJECT_ROOT / raw

        loss_payload = payload.get("loss", {})
        scheduler_payload = payload.get("scheduler", {})
        stopping_payload = payload.get("early_stopping", {})
        metrics = tuple(
            str(item)
            for item in payload.get("validation_metrics", ("psnr", "ssim", "lpips"))
        )
        resume_raw = payload.get("resume")
        resume = path_value("resume") if resume_raw else None
        return cls(
            train_manifest=path_value("train_manifest"),
            validation_manifest=path_value("validation_manifest"),
            data_root=path_value("data_root"),
            pretrained_weights=path_value("pretrained_weights"),
            output_dir=path_value("output_dir"),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "cuda")),
            amp=bool(payload.get("amp", True)),
            batch_size=int(payload.get("batch_size", 8)),
            num_workers=int(payload.get("num_workers", 4)),
            crop_size=int(payload.get("crop_size", 384)),
            darkening_probability=float(payload.get("darkening_probability", 0.5)),
            scale_factor=int(payload.get("scale_factor", 1)),
            epochs=int(payload.get("epochs", 30)),
            learning_rate=float(payload.get("learning_rate", 2e-5)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            grad_clip_norm=float(payload.get("grad_clip_norm", 0.1)),
            metrics_device=str(payload.get("metrics_device", "cuda")),
            validation_metrics=metrics,
            max_train_images=payload.get("max_train_images"),
            max_val_images=payload.get("max_val_images"),
            log_every=int(payload.get("log_every", 10)),
            save_every=int(payload.get("save_every", 1)),
            resume=resume,
            loss=LossWeights(**loss_payload),
            scheduler=SchedulerSettings(**scheduler_payload),
            early_stopping=EarlyStoppingSettings(**stopping_payload),
        )

    @classmethod
    def from_file(cls, path: Path) -> TrainConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, base=path.parent)
