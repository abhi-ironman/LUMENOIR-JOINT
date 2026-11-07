"""Experimental DL-primary hybrid correction.

The deep-learning candidate is the base image. Classical CV contributes only a
bounded high-frequency luminance correction, and only where its local detail is
closer to the source detail than the DL candidate is. A local-noise confidence
gate suppresses corrections in noisy regions.

This strategy is opt-in via LUMENOIR_HYBRID_STRATEGY=dl_primary. The legacy
LumenFuse algorithm remains the default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from ..classical_cv.arrays import as_float_rgb


@dataclass(frozen=True)
class DLPrimaryFusionConfig:
    """Conservative bounds for the experimental detail-transfer strategy."""

    detail_sigma: float = 1.0
    evidence_floor: float = 0.002
    evidence_span: float = 0.012
    detail_scale: float = 0.02
    noise_limit: float = 0.04
    correction_strength: float = 0.20
    max_correction: float = 0.015
    min_luma: float = 0.08
    max_relative_scale: float = 0.10


@dataclass(frozen=True)
class DLPrimaryFusionResult:
    """DL-primary image and diagnostics maps."""

    image: object
    confidence: object
    correction: object

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": "dl_primary",
            "shape": list(self.image.shape),
            "active_fraction": _mean(self.confidence > 0.05),
            "mean_confidence": _mean(self.confidence),
            "mean_absolute_luma_correction": _mean_abs(self.correction),
            "max_absolute_luma_correction": _max_abs(self.correction),
        }


def _mean(value) -> float:
    if hasattr(value, "detach"):
        tensor = value.detach()
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return float(tensor.mean().item())
    return float(np.mean(value))


def _mean_abs(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().abs().mean().item())
    return float(np.mean(np.abs(value)))


def _max_abs(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().abs().max().item())
    return float(np.max(np.abs(value)))


def _validate_config(config: DLPrimaryFusionConfig) -> None:
    positive = (
        config.detail_sigma,
        config.evidence_span,
        config.detail_scale,
        config.noise_limit,
        config.min_luma,
        config.max_relative_scale,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("DL-primary fusion scales must be finite and positive.")
    bounded = (
        config.evidence_floor,
        config.correction_strength,
        config.max_correction,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in bounded):
        raise ValueError("DL-primary fusion strengths must be finite and non-negative.")
    if config.max_relative_scale >= 1.0:
        raise ValueError("max_relative_scale must be below 1.0.")


def _luma(rgb, ops):
    if ops is not None:
        return ops.luma(rgb)
    return np.ascontiguousarray(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2],
        dtype=np.float32,
    )


def _blur(image, sigma: float, ops):
    if ops is not None:
        return ops.gaussian_blur(image, sigma)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma))


def _abs(value, ops):
    return ops.abs(value) if ops is not None else np.abs(value)


def _clip(value, low: float, high: float, ops):
    return ops.clip(value, low, high) if ops is not None else np.clip(value, low, high)


def _maximum(value, floor: float, ops):
    return ops.maximum(value, floor) if ops is not None else np.maximum(value, floor)


def fuse_dl_primary(
    original,
    classical,
    deep_learning,
    *,
    noise_map,
    ops=None,
    config: DLPrimaryFusionConfig | None = None,
) -> DLPrimaryFusionResult:
    """Preserve DL as the base and transfer only source-supported CV detail.

    The confidence mask is high only when (1) classical high-frequency luma
    detail is closer to the source detail than the DL detail, (2) the source
    contains measurable local detail, and (3) local estimated noise is below
    the configured limit. The applied luminance correction is clipped and then
    converted to a bounded RGB scale, preserving the DL candidate's chromatic
    direction except where output clipping is reached.

    With no eligible evidence the output is exactly the DL candidate.
    """
    config = config or DLPrimaryFusionConfig()
    _validate_config(config)

    if ops is None:
        source = as_float_rgb(original, name="original")
        cv = as_float_rgb(classical, name="classical")
        dl = as_float_rgb(deep_learning, name="deep_learning")
        local_noise = np.asarray(noise_map, dtype=np.float32)
        if not np.isfinite(local_noise).all() or np.any(local_noise < 0.0):
            raise ValueError("noise_map must contain finite non-negative values.")
    else:
        source = ops.ensure(original)
        cv = ops.ensure(classical)
        dl = ops.ensure(deep_learning)
        local_noise = ops.ensure(noise_map)
        torch = ops._torch
        if (
            not bool(torch.isfinite(source).all())
            or not bool(torch.isfinite(cv).all())
            or not bool(torch.isfinite(dl).all())
        ):
            raise ValueError("Fusion candidates contain NaN or infinite values.")
        if (
            not bool(torch.isfinite(local_noise).all())
            or bool((local_noise < 0.0).any())
        ):
            raise ValueError("noise_map must contain finite non-negative values.")
        candidates = (
            ("original", source),
            ("classical", cv),
            ("deep_learning", dl),
        )
        for name, candidate in candidates:
            if bool((candidate < 0.0).any()) or bool((candidate > 1.0).any()):
                raise ValueError(f"{name} must lie in [0, 1].")

    shape = tuple(source.shape)
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Fusion candidates must be HxWx3 RGB, got {shape}.")
    if tuple(cv.shape) != shape or tuple(dl.shape) != shape:
        raise ValueError("All fusion candidates must have identical HxWx3 shapes.")
    if tuple(local_noise.shape) != shape[:2]:
        raise ValueError(
            f"noise_map shape {tuple(local_noise.shape)} does not match {shape[:2]}."
        )

    source_y = _luma(source, ops)
    cv_y = _luma(cv, ops)
    dl_y = _luma(dl, ops)

    source_detail = source_y - _blur(source_y, config.detail_sigma, ops)
    cv_detail = cv_y - _blur(cv_y, config.detail_sigma, ops)
    dl_detail = dl_y - _blur(dl_y, config.detail_sigma, ops)

    # Positive evidence means the CV detail is closer to the observed source.
    cv_error = _abs(cv_detail - source_detail, ops)
    dl_error = _abs(dl_detail - source_detail, ops)
    evidence = dl_error - cv_error
    evidence_confidence = _clip(
        (evidence - config.evidence_floor) / config.evidence_span,
        0.0,
        1.0,
        ops,
    )
    source_detail_support = _clip(
        _abs(source_detail, ops) / config.detail_scale, 0.0, 1.0, ops
    )
    noise_confidence = _clip(
        1.0 - local_noise / config.noise_limit, 0.0, 1.0, ops
    )
    confidence = evidence_confidence * source_detail_support * noise_confidence

    detail_residual = cv_detail - dl_detail
    correction = _clip(
        config.correction_strength * confidence * detail_residual,
        -config.max_correction,
        config.max_correction,
        ops,
    )

    # Convert a small luminance residual to a bounded multiplicative RGB
    # correction so the DL candidate remains the chromatic base.
    relative = correction / _maximum(dl_y, config.min_luma, ops)
    scale = _clip(
        1.0 + relative,
        1.0 - config.max_relative_scale,
        1.0 + config.max_relative_scale,
        ops,
    )
    image = _clip(dl * scale[..., None], 0.0, 1.0, ops)

    if ops is None:
        image = np.ascontiguousarray(image.astype(np.float32))
        confidence = np.ascontiguousarray(confidence.astype(np.float32))
        correction = np.ascontiguousarray(correction.astype(np.float32))

    return DLPrimaryFusionResult(
        image=image,
        confidence=confidence,
        correction=correction,
    )


__all__ = [
    "DLPrimaryFusionConfig",
    "DLPrimaryFusionResult",
    "fuse_dl_primary",
]
