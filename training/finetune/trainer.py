"""Reproducible Zero-DCE++ fine-tuning on Kaggle.

The released checkpoint is opened read-only. Fine-tuned weights are written as
a bare ``state_dict`` so the existing strict inference wrapper can load them.
Training state (optimizer, scheduler, AMP scaler, history) lives beside it and
is never placed in ``backend/weights``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from backend.deep_learning.zero_dce_plus_plus import ZeroDCEPlusPlusModel
from backend.inference.device import resolve_device

from .config import TrainConfig
from .data import PairedDataset, load_train_validation_pairs, make_loader
from .losses import zero_dce_loss
from .metrics import (
    aggregate_metrics,
    evaluate_network,
    validation_loss,
    validation_objective,
)

LOGGER = logging.getLogger("lumenoir.finetune")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_network(checkpoint: Path, *, scale_factor: int, device: torch.device) -> torch.nn.Module:
    """Load through the existing strict wrapper, then detach its trainable net."""
    wrapper = ZeroDCEPlusPlusModel(checkpoint, device="cpu", scale_factor=scale_factor)
    wrapper.load()
    net = wrapper._net
    if net is None:  # pragma: no cover - defensive
        raise RuntimeError("Zero-DCE++ wrapper did not expose a network.")
    net.to(device)
    return net


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        raise RuntimeError(f"No finite {key} values were produced.")
    return float(np.mean(values))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


class FineTuner:
    def __init__(self, config: TrainConfig) -> None:
        config.validate()
        _seed_everything(config.seed)
        self.config = config
        self.device = torch.device(resolve_device(config.device))
        if config.device == "cuda" and self.device.type != "cuda":
            raise RuntimeError("CUDA was requested but is not available.")
        self.amp = bool(config.amp and self.device.type == "cuda")
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.resume is None and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {self.output_dir}. "
                "Use a new directory, or pass --resume with last_training.pt."
            )
        self.log_path = self.output_dir / "train.log"
        self._configure_logging()

        self.train_pairs, all_validation_pairs = load_train_validation_pairs(
            config.train_manifest,
            config.validation_manifest,
            config.data_root,
        )
        if config.max_train_images is not None:
            self.train_pairs = self.train_pairs[: config.max_train_images]
        if config.max_val_images is not None:
            self.validation_pairs = all_validation_pairs[: config.max_val_images]
            self.validation_metric_pairs = [
                pair for pair in all_validation_pairs if pair.has_reference
            ][: config.max_val_images]
        else:
            self.validation_pairs = all_validation_pairs
            self.validation_metric_pairs = [
                pair for pair in all_validation_pairs if pair.has_reference
            ]
        if not self.train_pairs or not self.validation_pairs:
            raise ValueError("Training and validation splits must both be non-empty.")
        if not self.validation_metric_pairs:
            raise ValueError(
                "Validation needs at least one paired LOL record for PSNR/SSIM/LPIPS; "
                "single-image records contribute only zero-reference validation loss."
            )
        self.validation_zero_reference_pairs = [
            pair for pair in self.validation_pairs if not pair.has_reference
        ]

        self.train_dataset = PairedDataset(
            self.train_pairs,
            crop_size=config.crop_size,
            train=True,
            seed=config.seed,
            darkening_probability=config.darkening_probability,
        )
        self.validation_dataset = PairedDataset(
            self.validation_pairs,
            train=False,
            seed=config.seed,
        )

        self.net = _load_network(
            config.pretrained_weights,
            scale_factor=config.scale_factor,
            device=self.device,
        )
        self.pretrained_net = _load_network(
            config.pretrained_weights,
            scale_factor=config.scale_factor,
            device=self.device,
        )
        self.pretrained_net.eval()
        for parameter in self.pretrained_net.parameters():
            parameter.requires_grad_(False)

        self.optimizer = Adam(
            self.net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.scheduler = self._build_scheduler()
        self.history: list[dict[str, Any]] = []
        self.start_epoch = 0
        self.best_objective = math.inf
        self.best_epoch = -1
        self.patience = 0
        self.baseline_rows: list[dict[str, Any]] = []
        self.last_rows: list[dict[str, Any]] = []
        self.baseline_metrics: dict[str, float] = {}
        self.best_metrics: dict[str, float] = {}

        if config.resume is not None:
            self._resume(config.resume)
        self._write_run_config()
        self._baseline_validation()
        self._write_history()

    def _configure_logging(self) -> None:
        LOGGER.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        for handler in list(LOGGER.handlers):
            if isinstance(handler, logging.FileHandler):
                LOGGER.removeHandler(handler)
                handler.close()
        if not any(isinstance(handler, logging.StreamHandler) for handler in LOGGER.handlers):
            LOGGER.addHandler(stream)
        LOGGER.addHandler(file_handler)

    def _build_scheduler(self) -> ReduceLROnPlateau | None:
        settings = self.config.scheduler
        if settings.name.lower() in {"none", "off", "disabled"}:
            return None
        if settings.name != "reduce_on_plateau":
            raise ValueError(f"Unsupported scheduler: {settings.name!r}")
        return ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=settings.factor,
            patience=settings.patience,
            min_lr=settings.min_lr,
        )

    def _write_run_config(self) -> None:
        payload = self.config.as_dict()
        payload["resolved_device"] = str(self.device)
        payload["amp_enabled"] = self.amp
        payload["torch_version"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device"] = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
        )
        (self.output_dir / "run_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _baseline_validation(self) -> None:
        keys = tuple(dict.fromkeys((*self.config.validation_metrics, "new_clipping")))
        self.baseline_rows = evaluate_network(
            self.pretrained_net,
            self.validation_metric_pairs,
            keys=keys,
            device=self.device,
            amp=self.amp,
            stage="pretrained",
        )
        rows = _aggregate_with_means(self.baseline_rows)
        fields = [
            "dataset",
            "stage",
            "count",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "lpips_mean",
            "lpips_std",
            "new_clipping_mean",
            "new_clipping_std",
        ]
        _write_csv(self.output_dir / "baseline_metrics.csv", rows, fields)
        self._baseline_summary = rows
        self.baseline_metrics = {
            key: _mean_metric(self.baseline_rows, key)
            for key in ("psnr", "ssim", "lpips", "new_clipping")
        }

    def _resume(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "state_dict" not in state:
            raise ValueError("Resume checkpoint must contain a training state_dict.")
        self.net.load_state_dict(state["state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = int(state.get("epoch", -1)) + 1
        self.best_objective = float(state.get("best_objective", math.inf))
        self.best_epoch = int(state.get("best_epoch", -1))
        self.best_metrics = dict(state.get("best_metrics", {}))
        self.patience = int(state.get("patience", 0))
        self.history = list(state.get("history", []))
        LOGGER.info("Resumed from %s at epoch %d", path, self.start_epoch)

    def _save_best(self, epoch: int, objective: float) -> None:
        state = {key: value.detach().cpu() for key, value in self.net.state_dict().items()}
        torch.save(state, self.output_dir / "best.pth")
        LOGGER.info("Saved best checkpoint at epoch %d (objective %.6f)", epoch, objective)

    def _save_last(self, epoch: int, objective: float) -> None:
        state = {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.net.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_objective": self.best_objective,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "patience": self.patience,
            "objective": objective,
            "history": self.history,
            "config": self.config.as_dict(),
        }
        torch.save(state, self.output_dir / "last_training.pt")

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(
            json.dumps(
                {
                    "baseline": self._baseline_summary,
                    "epochs": self.history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if self.history:
            fields = sorted({key for row in self.history for key in row})
            _write_csv(self.output_dir / "history.csv", self.history, fields)

    def _train_epoch(self, epoch: int) -> float:
        self.net.train()
        self.train_dataset.set_epoch(epoch)
        loader = make_loader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            epoch=epoch,
            pin_memory=self.device.type == "cuda",
        )
        total_loss = 0.0
        total_images = 0
        for step, batch in enumerate(loader, start=1):
            low = batch["low"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.amp,
            ):
                output, alpha = self.net(low)
                breakdown = zero_dce_loss(output, alpha, low, self.config.loss)
                loss = breakdown.total
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.net.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_size = low.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_images += batch_size
            if step % self.config.log_every == 0:
                LOGGER.info(
                    "epoch %d step %d/%d loss %.6f",
                    epoch + 1,
                    step,
                    len(loader),
                    float(loss.detach().cpu()),
                )
        return total_loss / max(1, total_images)

    def _validate(self, epoch: int) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
        val_loss = validation_loss(
            self.net,
            self.validation_zero_reference_pairs or self.validation_pairs,
            weights=self.config.loss,
            device=self.device,
            amp=self.amp,
            max_images=None,
        )
        keys = tuple(dict.fromkeys((*self.config.validation_metrics, "new_clipping")))
        rows = evaluate_network(
            self.net,
            self.validation_metric_pairs,
            keys=keys,
            device=self.device,
            amp=self.amp,
            stage="fine_tuned",
        )
        metrics = {key: _mean_metric(rows, key) for key in keys}
        self.last_rows = rows
        return val_loss, metrics, rows

    def fit(self) -> dict[str, Any]:
        LOGGER.info(
            "Starting fine-tuning: train=%d validation=%d device=%s amp=%s",
            len(self.train_pairs),
            len(self.validation_pairs),
            self.device,
            self.amp,
        )
        stopped_early = False
        for epoch in range(self.start_epoch, self.config.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, metrics, rows = self._validate(epoch)
            objective = validation_objective(metrics)
            if self.scheduler is not None:
                self.scheduler.step(objective)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            improved = objective < self.best_objective - self.config.early_stopping.min_delta
            if improved:
                self.best_objective = objective
                self.best_epoch = epoch
                self.best_metrics = dict(metrics)
                self.patience = 0
                self._save_best(epoch, objective)
            else:
                self.patience += 1
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "objective": objective,
                "best_objective": self.best_objective,
                "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
                "patience": self.patience,
                "learning_rate": learning_rate,
                **{f"{key}_mean": value for key, value in metrics.items()},
            }
            self.history.append(record)
            self._write_history()
            self._write_validation_rows(rows)
            LOGGER.info(
                "epoch %d/%d train_loss %.6f val_loss %.6f objective %.6f "
                "psnr %.4f ssim %.4f lpips %.4f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                objective,
                metrics.get("psnr", math.nan),
                metrics.get("ssim", math.nan),
                metrics.get("lpips", math.nan),
            )
            if epoch % self.config.save_every == 0 or improved:
                self._save_last(epoch, objective)
            if self.patience >= self.config.early_stopping.patience:
                LOGGER.info("Early stopping at epoch %d", epoch + 1)
                stopped_early = True
                break
        self._save_last(self.history[-1]["epoch"] - 1, self.history[-1]["objective"])
        summary = {
            "epochs_completed": len(self.history),
            "best_epoch": self.best_epoch + 1 if self.best_epoch >= 0 else None,
            "best_objective": self.best_objective,
            "stopped_early": stopped_early,
            "train_images": len(self.train_pairs),
            "validation_images": len(self.validation_pairs),
            "validation_paired_images": len(self.validation_metric_pairs),
            "validation_zero_reference_images": len(self.validation_zero_reference_pairs),
            "train_paired_images": sum(pair.has_reference for pair in self.train_pairs),
            "train_zero_reference_images": sum(
                not pair.has_reference for pair in self.train_pairs
            ),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self._write_comparison()
        (self.output_dir / "summary.md").write_text(
            self._summary_markdown(summary), encoding="utf-8"
        )
        return summary

    def _write_comparison(self) -> None:
        rows = [
            {
                "metric": key,
                "pretrained": self.baseline_metrics.get(key),
                "best_finetuned": self.best_metrics.get(key),
            }
            for key in ("psnr", "ssim", "lpips", "new_clipping")
        ]
        _write_csv(
            self.output_dir / "validation_comparison.csv",
            rows,
            ["metric", "pretrained", "best_finetuned"],
        )

    def _write_validation_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = [
            "dataset",
            "image_id",
            "stage",
            *[
                key
                for key in rows[0]
                if key not in {"dataset", "image_id", "stage"}
            ],
        ]
        _write_csv(self.output_dir / "validation_metrics.csv", rows, fields)

    def _summary_markdown(self, summary: dict[str, Any]) -> str:
        lines = [
            "# Zero-DCE++ fine-tuning",
            "",
            f"- Train images: {summary['train_images']}",
            f"- Validation records: {summary['validation_images']}",
            f"- Validation paired metric images: {summary['validation_paired_images']}",
            f"- Validation zero-reference images: {summary['validation_zero_reference_images']}",
            f"- Training paired inputs: {summary['train_paired_images']}",
            f"- Training zero-reference inputs: {summary['train_zero_reference_images']}",
            f"- Epochs completed: {summary['epochs_completed']}",
            f"- Best epoch: {summary['best_epoch']}",
            f"- Best validation objective: {summary['best_objective']:.6f}",
            f"- Early stopped: {summary['stopped_early']}",
            "- Validation uses the explicit validation manifest; test images were not used.",
            "- `best.pth` is the bare fine-tuned state dict; `last_training.pt` is resumable.",
            "",
            "## Baseline and fine-tuned validation",
            "",
            "| Metric | Pretrained | Best fine-tuned |",
            "| --- | ---: | ---: |",
        ]
        for key in ("psnr", "ssim", "lpips", "new_clipping"):
            lines.append(
                f"| {key} | {self.baseline_metrics.get(key, math.nan):.6f} | "
                f"{self.best_metrics.get(key, math.nan):.6f} |"
            )
        lines.extend(
            [
                "",
                f"Baseline rows: {len(self.baseline_rows)}; fine-tuned rows: {len(self.last_rows)}",
                "",
                "See `baseline_metrics.csv`, `validation_comparison.csv`, `history.csv`, "
                "`validation_metrics.csv` and `run_config.json`.",
                "",
            ]
        )
        return "\n".join(lines)


def _aggregate_with_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = aggregate_metrics(rows)
    return aggregates
