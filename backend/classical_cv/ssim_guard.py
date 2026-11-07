"""SSIM-based do-no-harm guard for the final output blend.

The live guard follows final_decision.md section 5.8. It uses the same 7x7,
per-channel SSIM definition as the evaluation harness and is only activated
for non-dark inputs whose proposed output falls below the SSIM trigger.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_WINDOW = 7
_RADIUS = _WINDOW // 2
_SAMPLES = float(_WINDOW * _WINDOW)
_COVARIANCE_NORMALISATION = _SAMPLES / (_SAMPLES - 1.0)
_C1 = 0.01**2
_C2 = 0.03**2

_TRIGGER_SSIM = 0.85
_TARGET_SSIM = 0.90
# Leave room for final uint8 encoding so the delivered image still meets 0.90.
_SEARCH_TARGET_SSIM = 0.915
_BINARY_SEARCH_STEPS = 12


def _validate_pair(source: Any, output: Any, ops: Any | None) -> None:
    if ops is not None:
        torch = ops._torch
        if not torch.is_tensor(source) or not torch.is_tensor(output):
            raise TypeError("Tensor SSIM requires source and output tensors.")
        if source.ndim != 3 or output.ndim != 3 or source.shape[-1] != 3:
            raise ValueError("SSIM expects HxWx3 RGB images.")
        if tuple(source.shape) != tuple(output.shape):
            raise ValueError(
                f"Source shape {tuple(source.shape)} differs from output "
                f"shape {tuple(output.shape)}."
            )
        height, width = int(source.shape[0]), int(source.shape[1])
    else:
        source_array = np.asarray(source)
        output_array = np.asarray(output)
        if source_array.ndim != 3 or output_array.ndim != 3:
            raise ValueError("SSIM expects HxWx3 RGB images.")
        if source_array.shape[-1] != 3 or output_array.shape[-1] != 3:
            raise ValueError("SSIM expects exactly three RGB channels.")
        if source_array.shape != output_array.shape:
            raise ValueError(
                f"Source shape {source_array.shape} differs from output "
                f"shape {output_array.shape}."
            )
        height, width = source_array.shape[:2]

    if min(height, width) < _WINDOW:
        raise ValueError(
            f"SSIM needs at least {_WINDOW} pixels per side, got {(height, width)}."
        )


def ssim_rgb(source: Any, output: Any, *, ops: Any | None = None) -> float:
    """Return mean channel-wise 7x7 SSIM for RGB float images in [0, 1].

    The implementation matches skimage.metrics.structural_similarity with
    channel_axis=2 and data_range=1.0 (uniform 7x7 window, sample covariance
    correction), without importing the evaluation-only package. The optional
    ops argument selects the existing tensor backend.
    """
    _validate_pair(source, output, ops)

    if ops is None:
        reference = np.asarray(source, dtype=np.float32)
        candidate = np.asarray(output, dtype=np.float32)
        channel_scores: list[float] = []
        for channel in range(3):
            x = reference[..., channel]
            y = candidate[..., channel]
            ux = cv2.boxFilter(
                x, -1, (_WINDOW, _WINDOW), normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            uy = cv2.boxFilter(
                y, -1, (_WINDOW, _WINDOW), normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            uxx = cv2.boxFilter(
                x * x, -1, (_WINDOW, _WINDOW), normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            uyy = cv2.boxFilter(
                y * y, -1, (_WINDOW, _WINDOW), normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            uxy = cv2.boxFilter(
                x * y, -1, (_WINDOW, _WINDOW), normalize=True,
                borderType=cv2.BORDER_REFLECT_101,
            )
            vx = _COVARIANCE_NORMALISATION * (uxx - ux * ux)
            vy = _COVARIANCE_NORMALISATION * (uyy - uy * uy)
            vxy = _COVARIANCE_NORMALISATION * (uxy - ux * uy)
            numerator = (2.0 * ux * uy + _C1) * (2.0 * vxy + _C2)
            denominator = (ux * ux + uy * uy + _C1) * (vx + vy + _C2)
            score = numerator / np.maximum(denominator, 1e-12)
            channel_scores.append(
                float(score[_RADIUS:-_RADIUS, _RADIUS:-_RADIUS].mean())
            )
        return float(np.mean(channel_scores))

    torch = ops._torch
    reference = source.to(dtype=torch.float32)
    candidate = output.to(dtype=torch.float32)
    channel_scores = []
    for channel in range(3):
        x = reference[..., channel]
        y = candidate[..., channel]
        ux = ops.box_filter(x, _RADIUS)
        uy = ops.box_filter(y, _RADIUS)
        uxx = ops.box_filter(x * x, _RADIUS)
        uyy = ops.box_filter(y * y, _RADIUS)
        uxy = ops.box_filter(x * y, _RADIUS)
        vx = _COVARIANCE_NORMALISATION * (uxx - ux * ux)
        vy = _COVARIANCE_NORMALISATION * (uyy - uy * uy)
        vxy = _COVARIANCE_NORMALISATION * (uxy - ux * uy)
        numerator = (2.0 * ux * uy + _C1) * (2.0 * vxy + _C2)
        denominator = (ux * ux + uy * uy + _C1) * (vx + vy + _C2)
        score = numerator / torch.clamp(denominator, min=1e-12)
        channel_scores.append(
            score[_RADIUS:-_RADIUS, _RADIUS:-_RADIUS].mean()
        )
    return float(torch.stack(channel_scores).mean().item())


def _blend(source: Any, enhanced: Any, alpha: float, ops: Any | None):
    if ops is None:
        return np.ascontiguousarray(
            np.clip(
                (1.0 - alpha) * np.asarray(source, dtype=np.float32)
                + alpha * np.asarray(enhanced, dtype=np.float32),
                0.0,
                1.0,
            ).astype(np.float32)
        )
    torch = ops._torch
    source32 = source.to(dtype=torch.float32)
    enhanced32 = enhanced.to(dtype=torch.float32)
    return torch.clamp(
        (1.0 - alpha) * source32 + alpha * enhanced32, 0.0, 1.0
    )


def blend_with_ssim_guard(
    source: Any,
    enhanced: Any,
    *,
    alpha: float,
    input_median_luminance: float,
    ops: Any | None = None,
) -> tuple[Any, dict[str, float | bool | str | None]]:
    """Blend toward the source when a non-dark output violates the SSIM guard.

    The section 5.8 trigger remains strict: guard only when p50 >= 0.25 and
    proposed SSIM < 0.85. Once triggered, reduce alpha to target SSIM >= 0.915
    internally, leaving encoding margin for the delivered-image floor of 0.90.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    proposed = _blend(source, enhanced, alpha, ops)
    diagnostics: dict[str, float | bool | str | None] = {
        "applied": False,
        "reason": "",
        "initial_ssim": None,
        "final_ssim": None,
        "alpha_before": alpha,
        "alpha_after": alpha,
        "trigger_ssim": _TRIGGER_SSIM,
        "target_ssim": _TARGET_SSIM,
        "search_target_ssim": _SEARCH_TARGET_SSIM,
    }

    if input_median_luminance < 0.25:
        diagnostics["reason"] = "dark_input"
        return proposed, diagnostics
    if alpha <= 0.0:
        diagnostics["reason"] = "blend_disabled"
        return proposed, diagnostics

    initial_score = ssim_rgb(source, proposed, ops=ops)
    diagnostics["initial_ssim"] = initial_score
    diagnostics["final_ssim"] = initial_score
    if initial_score >= _TRIGGER_SSIM:
        diagnostics["reason"] = "initial_ssim_above_trigger"
        return proposed, diagnostics

    low = 0.0
    high = alpha
    accepted_alpha = 0.0
    for _ in range(_BINARY_SEARCH_STEPS):
        midpoint = (low + high) / 2.0
        trial = _blend(source, enhanced, midpoint, ops)
        score = ssim_rgb(source, trial, ops=ops)
        if score >= _SEARCH_TARGET_SSIM:
            low = midpoint
            accepted_alpha = midpoint
        else:
            high = midpoint

    final = _blend(source, enhanced, accepted_alpha, ops)
    final_score = ssim_rgb(source, final, ops=ops)

    # Be conservative if SSIM is locally non-monotonic or numerical rounding
    # places the selected blend below the delivery floor.
    while final_score < _TARGET_SSIM and accepted_alpha > 1e-6:
        accepted_alpha *= 0.5
        final = _blend(source, enhanced, accepted_alpha, ops)
        final_score = ssim_rgb(source, final, ops=ops)

    if final_score < _TARGET_SSIM:
        accepted_alpha = 0.0
        final = _blend(source, enhanced, 0.0, ops)
        final_score = ssim_rgb(source, final, ops=ops)

    diagnostics.update(
        {
            "applied": True,
            "reason": "reduced_blend_to_preserve_structure",
            "final_ssim": final_score,
            "alpha_after": accepted_alpha,
        }
    )
    return final, diagnostics


__all__ = ["blend_with_ssim_guard", "ssim_rgb"]
