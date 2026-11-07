"""Validation metrics and loss aggregation for fine-tuning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from backend.evaluation.aggregate import summarise
from backend.evaluation.harness import evaluate_arrays
from training.data_pipeline.config import SICE_LONG_EDGE
from training.data_pipeline.preprocessing import load_rgb_float32, resize_long_edge

from .config import LossWeights
from .data import PairRecord
from .losses import zero_dce_loss


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().clamp(0.0, 1.0).numpy()
    return np.ascontiguousarray(array.transpose(1, 2, 0))


def _autocast(device: torch.device, amp: bool):
    if device.type == "cuda" and amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def evaluate_network(
    net: torch.nn.Module,
    pairs: list[PairRecord],
    *,
    keys: Iterable[str] = ("psnr", "ssim", "lpips", "new_clipping"),
    device: torch.device,
    amp: bool = True,
    max_images: int | None = None,
    stage: str = "fine_tuned",
) -> list[dict[str, Any]]:
    """Evaluate every pair and return per-image metric rows."""
    net.eval()
    rows: list[dict[str, Any]] = []
    selected = pairs if max_images is None else pairs[:max_images]
    with torch.no_grad():
        for pair in selected:
            if pair.high is None:
                raise ValueError(
                    f"Reference metrics require a paired record; {pair.image_id} has no high image."
                )
            low_np = load_rgb_float32(pair.low)
            high_np = load_rgb_float32(pair.high)
            low_tensor = torch.from_numpy(
                np.ascontiguousarray(low_np.transpose(2, 0, 1))
            ).unsqueeze(0).to(device)
            with _autocast(device, amp):
                output, _alpha = net(low_tensor)
            metrics = evaluate_arrays(
                tensor_to_rgb(output[0]),
                reference=high_np,
                original=low_np,
                keys=keys,
            )
            missing = [key for key, value in metrics.items() if value.value is None]
            if missing:
                details = "; ".join(f"{key}: {metrics[key].notes}" for key in missing)
                raise RuntimeError(f"Validation metric unavailable ({details}).")
            row: dict[str, Any] = {
                "dataset": pair.dataset,
                "image_id": pair.image_id,
                "stage": stage,
            }
            row.update({key: metric.value for key, metric in metrics.items()})
            rows.append(row)
    return rows


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean/std/count for every metric, grouped by dataset and stage."""
    if not rows:
        return []
    metric_keys = sorted(
        key
        for key in rows[0]
        if key not in {"dataset", "image_id", "stage"}
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["stage"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, stage), items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "dataset": dataset,
            "stage": stage,
            "count": len(items),
        }
        for key in metric_keys:
            values = [float(item[key]) for item in items if item.get(key) is not None]
            summary = summarise(values)
            if summary is not None:
                record[f"{key}_mean"] = summary.mean
                record[f"{key}_std"] = summary.std
        output.append(record)
    return output


def validation_objective(rows: dict[str, float]) -> float:
    """final_decision.md §2.6: lower is better."""
    required = {"lpips", "ssim", "new_clipping"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Validation objective needs {sorted(missing)}.")
    return (
        float(rows["lpips"])
        + 0.5 * (1.0 - float(rows["ssim"]))
        + 5.0 * max(0.0, float(rows["new_clipping"]))
    )


def validation_loss(
    net: torch.nn.Module,
    pairs: list[PairRecord],
    *,
    weights: LossWeights,
    device: torch.device,
    amp: bool = True,
    max_images: int | None = None,
) -> float:
    """Mean documented training loss on unaugmented validation images."""
    net.eval()
    selected = pairs if max_images is None else pairs[:max_images]
    values: list[float] = []
    with torch.no_grad():
        for pair in selected:
            low_np = load_rgb_float32(pair.low)
            if pair.dataset == "sice":
                low_np = resize_long_edge(low_np, SICE_LONG_EDGE)
            low_tensor = torch.from_numpy(
                np.ascontiguousarray(low_np.transpose(2, 0, 1))
            ).unsqueeze(0).to(device)
            with _autocast(device, amp):
                output, alpha = net(low_tensor)
                breakdown = zero_dce_loss(output, alpha, low_tensor, weights)
            values.append(float(breakdown.total.detach().cpu()))
    return float(np.mean(values)) if values else float("nan")
