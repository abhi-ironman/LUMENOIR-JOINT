"""GPU HueLock with optional source-aware chroma anchoring."""
from __future__ import annotations

from dataclasses import dataclass

from .hue_lock import HueLockConfig
from .scan import LumenScanResult
from .torch_colour import mid_tone_lab_stats_tensor


@dataclass(frozen=True)
class TorchHueLockResult:
    image: object
    cast_a: float
    cast_b: float
    cast_magnitude: float
    beta_raw: float
    beta_applied: float
    shift_magnitude: float
    spread: float
    spread_guarded: bool
    source_anchor_mean_weight: float = 0.0
    source_anchor_coverage: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.image.shape),
            "cast": [self.cast_a, self.cast_b],
            "cast_magnitude": self.cast_magnitude,
            "beta_raw": self.beta_raw,
            "beta_applied": self.beta_applied,
            "shift_magnitude": self.shift_magnitude,
            "spread": self.spread,
            "spread_guarded": self.spread_guarded,
            "source_anchor_mean_weight": self.source_anchor_mean_weight,
            "source_anchor_coverage": self.source_anchor_coverage,
        }


def _resize_rgb(rgb, max_edge: int, *, ops):
    torch = ops.torch
    height, width = rgb.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale >= 1.0:
        return rgb
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(nchw, size=size, mode="area")
    return resized[0].permute(1, 2, 0).contiguous()


def _median3(value, *, ops):
    torch = ops.torch
    functional = torch.nn.functional
    padded = functional.pad(value[None, None], (1, 1, 1, 1), mode="replicate")
    patches = functional.unfold(padded, kernel_size=3)
    return patches.median(dim=1).values.reshape(value.shape)


def _source_anchor_maps(reference, *, config, noise_sigma: float, output_shape, ops):
    torch = ops.torch
    functional = torch.nn.functional
    source_small = _resize_rgb(reference, config.analysis_max_edge, ops=ops)
    source_lab = ops.rgb_to_lab(source_small)
    source_a = _median3(source_lab[..., 1], ops=ops)
    source_b = _median3(source_lab[..., 2], ops=ops)
    chroma = torch.hypot(source_a, source_b)
    luma = (
        0.299 * source_small[..., 0]
        + 0.587 * source_small[..., 1]
        + 0.114 * source_small[..., 2]
    )
    luminance_confidence = torch.clamp((luma - 0.025) / 0.18, 0.0, 1.0)
    neutral_confidence = torch.exp(-torch.square(chroma / 18.0))
    noise_confidence = float(max(0.55, min(1.0, 1.0 - 12.0 * max(0.0, noise_sigma))))
    anchor_small = (
        float(config.source_anchor_strength)
        * luminance_confidence
        * neutral_confidence
        * noise_confidence
    )

    height, width = output_shape
    anchor = functional.interpolate(
        anchor_small[None, None], size=(height, width),
        mode="bilinear", align_corners=False,
    )[0, 0]
    source_a = functional.interpolate(
        source_a[None, None], size=(height, width),
        mode="bilinear", align_corners=False,
    )[0, 0]
    source_b = functional.interpolate(
        source_b[None, None], size=(height, width),
        mode="bilinear", align_corners=False,
    )[0, 0]
    return anchor, source_a, source_b


def correct_tensor(
    image,
    *,
    scan_result: LumenScanResult,
    config: HueLockConfig | None = None,
    reference_rgb=None,
    ops,
) -> TorchHueLockResult:
    config = config or HueLockConfig()
    torch = ops.torch
    rgb = image.to(ops.dtype)
    features = scan_result.features
    parameters = scan_result.parameters

    if not 0.0 <= config.source_anchor_strength <= 0.55:
        raise ValueError("source_anchor_strength must be in [0, 0.55]")
    if config.analysis_max_edge < 32:
        raise ValueError("analysis_max_edge must be at least 32")

    cast_a = float(features.colour_cast_a)
    cast_b = float(features.colour_cast_b)
    cast_magnitude = float(features.colour_cast_magnitude)
    beta_raw = float(parameters.hue_lock_beta)

    analysis_rgb = _resize_rgb(rgb, config.analysis_max_edge, ops=ops)
    stats = mid_tone_lab_stats_tensor(
        analysis_rgb, p50=float(features.p50), ops=ops
    )
    spread = max(stats.std_a, stats.std_b)
    spread_guarded = spread > config.spread_threshold

    beta = beta_raw / 2.0 if (spread_guarded and config.halve_on_spread) else beta_raw
    if cast_magnitude > 0.0:
        beta = min(beta, float(config.max_shift) / cast_magnitude)
    beta = max(0.0, beta)

    anchor_enabled = (
        reference_rgb is not None and config.source_anchor_strength > 0.0
    )
    if beta <= 0.0 and not anchor_enabled:
        return TorchHueLockResult(
            image=rgb,
            cast_a=cast_a,
            cast_b=cast_b,
            cast_magnitude=cast_magnitude,
            beta_raw=beta_raw,
            beta_applied=0.0,
            shift_magnitude=0.0,
            spread=spread,
            spread_guarded=spread_guarded,
        )

    lab = ops.rgb_to_lab(rgb)
    if beta > 0.0 and cast_magnitude > 0.0:
        lab[..., 1] = lab[..., 1] - float(beta * cast_a)
        lab[..., 2] = lab[..., 2] - float(beta * cast_b)

    anchor_mean = 0.0
    anchor_coverage = 0.0
    if anchor_enabled:
        if tuple(reference_rgb.shape) != tuple(rgb.shape):
            raise ValueError("reference_rgb must match the enhanced tensor shape")
        anchor, source_a, source_b = _source_anchor_maps(
            reference_rgb.to(ops.dtype),
            config=config,
            noise_sigma=float(features.noise_sigma),
            output_shape=rgb.shape[:2],
            ops=ops,
        )
        lab[..., 1] = (1.0 - anchor) * lab[..., 1] + anchor * source_a
        lab[..., 2] = (1.0 - anchor) * lab[..., 2] + anchor * source_b
        anchor_mean = float(anchor.mean())
        anchor_coverage = float((anchor > 0.05).to(torch.float32).mean())

    corrected = torch.clamp(ops.lab_to_rgb(lab), min=0.0, max=1.0).to(rgb.dtype)
    if corrected.shape != rgb.shape:
        raise RuntimeError(
            f"HueLock changed the image shape: {tuple(rgb.shape)} -> {tuple(corrected.shape)}."
        )
    return TorchHueLockResult(
        image=corrected,
        cast_a=cast_a,
        cast_b=cast_b,
        cast_magnitude=cast_magnitude,
        beta_raw=beta_raw,
        beta_applied=beta,
        shift_magnitude=beta * cast_magnitude,
        spread=spread,
        spread_guarded=spread_guarded,
        source_anchor_mean_weight=anchor_mean,
        source_anchor_coverage=anchor_coverage,
    )


__all__ = ["TorchHueLockResult", "correct_tensor"]
