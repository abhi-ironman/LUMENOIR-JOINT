"""Pipeline stages.

Each class is a thin orchestration layer over an existing module:

- Analysis → LumenScan
- Classical CV → LumenRetinex
- Deep learning → the registered Zero-DCE++ model through ModelManager
- Fusion → LumenFuse
- Denoising → EdgeGuard
- Colour correction → HueLock
- Quality assessment → the adaptive alpha blend with a non-dark SSIM guard

The stage classes contain no algorithm logic of their own. ``PendingStage``
remains available for custom/partial pipelines, but the default pipeline uses
the implemented stages only.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..classical_cv.edge_guard import denoise as edge_guard_denoise
from ..classical_cv.hue_lock import HueLockConfig, correct as hue_lock_correct
from ..classical_cv.retinex import LumenRetinexConfig, enhance as retinex_enhance
from ..classical_cv.scan import LumenScanResult, scan
from ..classical_cv.ssim_guard import blend_with_ssim_guard
from ..fusion.lumen_fuse import fuse as fuse_candidates
from ..inference.errors import InferenceFailedError, ModelNotAvailableError
from ..inference.manager import ModelManager
from ..runtime.config import PipelineRuntime
from .phase_engine import PhaseOverrides, apply_phase_overrides
from .types import (
    ArtifactKind,
    EnhancementMode,
    StageName,
    StageStatus,
)


class StageNotImplemented(NotImplementedError):
    """Raised by a stage that has no implementation yet."""


@dataclass
class StageContext:
    """Mutable state passed along the pipeline for a single image."""

    image: Image.Image
    mode: EnhancementMode
    original_size: tuple[int, int]
    outputs: dict[ArtifactKind, Image.Image] = field(default_factory=dict)

    rgb: np.ndarray | None = None
    scan_result: LumenScanResult | None = None
    bypass: bool = False
    classical: np.ndarray | None = None
    deep_learning: np.ndarray | None = None
    fused: np.ndarray | None = None
    enhanced: np.ndarray | None = None
    runtime: PipelineRuntime | None = None
    ops: object | None = None
    rgb_tensor: object | None = None
    classical_tensor: object | None = None
    deep_learning_tensor: object | None = None
    fused_tensor: object | None = None
    enhanced_tensor: object | None = None
    final_tensor: object | None = None
    model_manager: ModelManager | None = None
    capture_diagnostics: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)
    diagnostic_images: dict[str, Image.Image] = field(default_factory=dict)
    phase_overrides: PhaseOverrides = field(default_factory=PhaseOverrides)

    def publish(self, kind: ArtifactKind, image: Image.Image) -> None:
        """Record an intermediate result so it can be reported to the client."""
        self.outputs[kind] = image


class PipelineStage(ABC):
    """One step of the enhancement pipeline."""

    name: StageName
    description: str
    produces: tuple[ArtifactKind, ...] = ()

    @property
    @abstractmethod
    def status(self) -> StageStatus:
        """Whether this stage can run in the current build."""

    @abstractmethod
    def run(self, context: StageContext) -> StageContext:
        """Process the context and return it (stages may publish artifacts)."""


class ImplementedStage(PipelineStage):
    """Base class for a stage backed by a real module."""

    @property
    def status(self) -> StageStatus:
        return StageStatus.IMPLEMENTED


class PendingStage(PipelineStage):
    """A documented stage whose algorithm has not been decided yet."""

    def __init__(
        self,
        name: StageName,
        description: str,
        produces: tuple[ArtifactKind, ...] = (),
        pending_reason: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.produces = produces
        self.pending_reason = pending_reason or (
            "This stage is not implemented: its algorithm is pending research."
        )

    @property
    def status(self) -> StageStatus:
        return StageStatus.NOT_IMPLEMENTED

    def run(self, context: StageContext) -> StageContext:
        raise StageNotImplemented(self.pending_reason)


def _to_array(image: Image.Image, *, name: str = "image") -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise InferenceFailedError(f"{name} is not an RGB image.")
    return np.ascontiguousarray(array)


def _to_pil(array: np.ndarray, *, name: str = "output") -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise RuntimeError(f"{name} is not HxWx3, got {values.shape}.")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or infinite values.")
    encoded = np.clip(np.round(values * 255.0), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(encoded)


def _map_to_pil(array: np.ndarray, *, vmax: float, name: str) -> Image.Image:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise RuntimeError(f"{name} is not a 2-D map, got {values.shape}.")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains NaN or infinite values.")
    scale = float(vmax) if vmax > 0.0 else 1.0
    encoded = np.clip(values / scale, 0.0, 1.0)
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8))


def _retinex_config_from_environment(
    overrides: PhaseOverrides | None = None,
) -> LumenRetinexConfig:
    """Resolve shared CPU/GPU Retinex settings and retry shoulder changes."""
    overrides = overrides or PhaseOverrides()
    raw = os.getenv("LUMENOIR_RETINEX_DOWNSAMPLE", "2").strip()
    try:
        downsample = int(raw)
    except ValueError as exc:
        raise ValueError(
            "LUMENOIR_RETINEX_DOWNSAMPLE must be an integer from 1 to 4."
        ) from exc
    if not 1 <= downsample <= 4:
        raise ValueError("LUMENOIR_RETINEX_DOWNSAMPLE must be between 1 and 4.")
    shoulder_start = float(np.clip(0.85 + overrides.shoulder_start_delta, 0.55, 0.99))
    return LumenRetinexConfig(
        illumination_downsample=downsample,
        shoulder_start=shoulder_start,
    )


def _hue_lock_config_from_environment() -> HueLockConfig:
    """Resolve opt-in source-chroma anchoring; off by default."""
    raw = os.getenv("LUMENOIR_HUELOCK_SOURCE_ANCHOR", "0").strip()
    try:
        strength = float(raw or "0")
    except ValueError as exc:
        raise ValueError(
            "LUMENOIR_HUELOCK_SOURCE_ANCHOR must be between 0 and 0.55."
        ) from exc
    if not np.isfinite(strength) or not 0.0 <= strength <= 0.55:
        raise ValueError("LUMENOIR_HUELOCK_SOURCE_ANCHOR must be between 0 and 0.55.")
    return HueLockConfig(source_anchor_strength=strength)


def _require(value: object, name: str) -> None:
    if value is None:
        raise RuntimeError(f"Pipeline stage dependency '{name}' is missing.")


class AnalysisStage(ImplementedStage):
    name = StageName.ANALYSIS
    description = (
        "Measure illumination, dynamic range, colour cast and noise with LumenScan."
    )
    produces = (
        ArtifactKind.ORIGINAL,
        ArtifactKind.ILLUMINATION_MAP,
        ArtifactKind.NOISE_MAP,
    )

    def run(self, context: StageContext) -> StageContext:
        context.rgb = _to_array(context.image)
        if context.ops is not None:
            from ..classical_cv.torch_scan import scan_tensor

            context.rgb_tensor = context.ops.ensure(context.rgb)
            measured = scan_tensor(context.rgb_tensor, ops=context.ops)
        else:
            measured = scan(context.rgb)
        context.scan_result = apply_phase_overrides(
            measured, context.phase_overrides
        )
        context.bypass = bool(context.scan_result.parameters.bypass_gate)
        if context.capture_diagnostics:
            context.diagnostics["scan"] = context.scan_result.as_dict()

        context.publish(ArtifactKind.ORIGINAL, context.image.copy())
        illumination = (
            context.ops.to_numpy(context.scan_result.illumination)
            if context.ops is not None
            else context.scan_result.illumination
        )
        noise = (
            context.ops.to_numpy(context.scan_result.noise_map)
            if context.ops is not None
            else context.scan_result.noise_map
        )
        context.publish(
            ArtifactKind.ILLUMINATION_MAP,
            _map_to_pil(illumination, vmax=1.0, name="illumination"),
        )
        context.publish(
            ArtifactKind.NOISE_MAP,
            _map_to_pil(noise, vmax=max(float(noise.max()), 1e-6), name="noise map"),
        )
        return context


class ClassicalStage(ImplementedStage):
    name = StageName.CLASSICAL_CV
    description = "LumenRetinex adaptive illumination correction, CLAHE and shoulder."
    produces = (ArtifactKind.CLASSICAL_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")

        if context.bypass:
            context.classical = context.rgb
            context.enhanced = context.rgb
            if context.ops is not None:
                context.classical_tensor = context.rgb_tensor
                context.enhanced_tensor = context.rgb_tensor
            return context

        if context.ops is not None:
            from ..classical_cv.torch_retinex import enhance_tensor

            result = enhance_tensor(
                context.rgb_tensor,
                scan_result=context.scan_result,
                config=_retinex_config_from_environment(context.phase_overrides),
                ops=context.ops,
            )
            context.classical_tensor = result.image
            context.enhanced_tensor = result.image
            array = context.ops.to_numpy(result.image)
        else:
            result = retinex_enhance(
                context.rgb,
                scan_result=context.scan_result,
                config=_retinex_config_from_environment(context.phase_overrides),
            )
            context.classical = result.image
            context.enhanced = result.image
            array = result.image
        context.publish(ArtifactKind.CLASSICAL_OUTPUT, _to_pil(array))
        return context


class DeepLearningStage(ImplementedStage):
    name = StageName.DEEP_LEARNING
    description = "Zero-DCE++ inference through the load-once ModelManager."
    produces = (ArtifactKind.DEEP_LEARNING_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")

        if context.bypass:
            context.deep_learning = context.rgb
            context.enhanced = context.rgb
            if context.ops is not None:
                context.deep_learning_tensor = context.rgb_tensor
                context.enhanced_tensor = context.rgb_tensor
            return context

        manager = context.model_manager
        if manager is None:
            raise ModelNotAvailableError(
                "No inference model manager is configured for this pipeline."
            )

        if context.ops is not None:
            outcome = manager.infer_tensor(context.rgb_tensor)
            tensor = outcome.image
            strength = float(context.phase_overrides.dl_strength_scale)
            if strength < 1.0:
                tensor = (
                    context.rgb_tensor
                    + strength * (tensor - context.rgb_tensor)
                ).clamp(0.0, 1.0)
            if tuple(tensor.shape) != tuple(context.rgb.shape):
                raise InferenceFailedError(
                    f"The model returned {tuple(tensor.shape)}, expected "
                    f"{context.rgb.shape}."
                )
            if context.capture_diagnostics and outcome.curve_alpha is not None:
                alpha = outcome.curve_alpha
                context.diagnostics["zero_dce_curve_alpha_range"] = [
                    float(alpha.min()),
                    float(alpha.max()),
                ]
            context.deep_learning_tensor = tensor
            context.enhanced_tensor = tensor
            array = context.ops.to_numpy(tensor)
        else:
            outcome = manager.infer(context.image)
            try:
                array = _to_array(outcome.image, name="model output")
            finally:
                outcome.image.close()

            strength = float(context.phase_overrides.dl_strength_scale)
            if strength < 1.0:
                array = np.ascontiguousarray(
                    np.clip(
                        context.rgb + strength * (array - context.rgb),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                )
            if array.shape != context.rgb.shape:
                raise InferenceFailedError(
                    f"The model returned {array.shape}, expected {context.rgb.shape}."
                )

            context.deep_learning = array
            context.enhanced = array
        context.publish(ArtifactKind.DEEP_LEARNING_OUTPUT, _to_pil(array))
        return context


class FusionStage(ImplementedStage):
    name = StageName.FUSION
    description = "LumenFuse noise-aware Laplacian-pyramid fusion of I0, C and D."
    produces = (
        ArtifactKind.FUSION_OUTPUT,
        ArtifactKind.ENHANCEMENT_STRENGTH_MAP,
    )

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        if context.ops is not None:
            _require(context.rgb_tensor, "rgb_tensor")
            _require(context.classical_tensor, "classical_tensor")
            _require(context.deep_learning_tensor, "deep_learning_tensor")
        else:
            _require(context.classical, "classical")
            _require(context.deep_learning, "deep_learning")

        if context.bypass:
            context.fused = context.rgb
            context.enhanced = context.rgb
            if context.ops is not None:
                context.fused_tensor = context.rgb_tensor
                context.enhanced_tensor = context.rgb_tensor
            return context

        if context.ops is not None:
            from ..fusion.lumen_fuse import fuse_tensors

            result = fuse_tensors(
                context.rgb_tensor,
                context.classical_tensor,
                context.deep_learning_tensor,
                noise_map=context.scan_result.noise_map,
                parameters=context.scan_result.parameters,
                ops=context.ops,
                sigma=float(context.scan_result.features.noise_sigma),
            )
            context.fused_tensor = result.image
            context.enhanced_tensor = result.image
            array = context.ops.to_numpy(result.image)
            weights = context.ops.to_numpy(result.weights)
            if context.capture_diagnostics:
                context.diagnostics["fusion"] = {
                    "shape": list(array.shape),
                    "levels": result.levels,
                    "sigma": result.sigma,
                }
                context.diagnostics["fusion_weights"] = weights
        else:
            result = fuse_candidates(
                context.rgb,
                context.classical,
                context.deep_learning,
                scan_result=context.scan_result,
                backend=os.getenv("LUMENOIR_FUSE_BACKEND", "auto"),
                device=os.getenv("LUMENOIR_FUSE_DEVICE") or None,
                precision=os.getenv("LUMENOIR_FUSE_PRECISION", "float32"),
            )
            context.fused = result.image
            context.enhanced = result.image
            array = result.image
            weights = result.weights
            if context.capture_diagnostics:
                context.diagnostics["fusion"] = result.as_dict()
                context.diagnostics["fusion_weights"] = result.weights.copy()
                if result.initial_weights is not None:
                    context.diagnostics["fusion_weights_initial"] = (
                        result.initial_weights.copy()
                    )
        context.publish(ArtifactKind.FUSION_OUTPUT, _to_pil(array))
        strength = 1.0 - weights[..., 0]
        context.publish(
            ArtifactKind.ENHANCEMENT_STRENGTH_MAP,
            _map_to_pil(strength, vmax=1.0, name="enhancement strength"),
        )
        return context


class DenoisingStage(ImplementedStage):
    name = StageName.DENOISING
    description = "EdgeGuard gain-aware luma guided filter and chroma bilateral filter."

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        if context.ops is not None:
            _require(context.enhanced_tensor, "enhanced_tensor")
        else:
            _require(context.enhanced, "enhanced")

        if context.bypass:
            context.enhanced = context.rgb
            if context.ops is not None:
                context.enhanced_tensor = context.rgb_tensor
            return context

        if context.ops is not None:
            from ..classical_cv.torch_edge_guard import denoise_tensor

            sigma = (
                context.scan_result.features.noise_sigma
                * context.phase_overrides.denoise_strength_scale
            )
            result = denoise_tensor(
                context.enhanced_tensor,
                context.rgb_tensor,
                sigma=sigma,
                scan_result=context.scan_result,
                ops=context.ops,
            )
            context.enhanced_tensor = result.image
            array = context.ops.to_numpy(result.image)
        else:
            sigma = (
                context.scan_result.features.noise_sigma
                * context.phase_overrides.denoise_strength_scale
            )
            result = edge_guard_denoise(
                context.enhanced,
                context.rgb,
                sigma=sigma,
                scan_result=context.scan_result,
            )
            context.enhanced = result.image
            array = result.image
        if context.capture_diagnostics:
            context.diagnostics["edge_guard"] = result.as_dict()
            context.diagnostic_images["edge_guard"] = _to_pil(array)
        return context


class ColourCorrectionStage(ImplementedStage):
    name = StageName.COLOUR_CORRECTION
    description = "HueLock guarded LAB colour-cast correction."

    def run(self, context: StageContext) -> StageContext:
        _require(context.scan_result, "scan_result")
        if context.ops is not None:
            _require(context.enhanced_tensor, "enhanced_tensor")
        else:
            _require(context.enhanced, "enhanced")

        if context.bypass:
            return context

        if context.ops is not None:
            from ..classical_cv.torch_hue_lock import correct_tensor

            result = correct_tensor(
                context.enhanced_tensor,
                scan_result=context.scan_result,
                config=_hue_lock_config_from_environment(),
                reference_rgb=context.rgb_tensor,
                ops=context.ops,
            )
            context.enhanced_tensor = result.image
            array = context.ops.to_numpy(result.image)
        else:
            result = hue_lock_correct(
                context.enhanced,
                scan_result=context.scan_result,
                config=_hue_lock_config_from_environment(),
                reference_rgb=context.rgb,
            )
            context.enhanced = result.image
            array = result.image
        if context.capture_diagnostics:
            context.diagnostics["hue_lock"] = result.as_dict()
            context.diagnostic_images["hue_lock"] = _to_pil(array)
        return context


class QualityAssessmentStage(ImplementedStage):
    name = StageName.QUALITY_ASSESSMENT
    description = (
        "Adaptive alpha blend, non-dark SSIM protection and final output validation."
    )
    produces = (ArtifactKind.FINAL_OUTPUT,)

    def run(self, context: StageContext) -> StageContext:
        _require(context.rgb, "rgb")
        _require(context.scan_result, "scan_result")
        if context.ops is not None:
            _require(context.rgb_tensor, "rgb_tensor")
            _require(context.enhanced_tensor, "enhanced_tensor")
        else:
            _require(context.enhanced, "enhanced")

        alpha = (
            0.0
            if context.bypass
            else float(context.scan_result.parameters.output_blend_alpha)
        )
        if context.capture_diagnostics:
            context.diagnostics["output_blend_alpha"] = alpha

        if context.bypass:
            if context.ops is not None:
                final_tensor = context.rgb_tensor
                context.final_tensor = final_tensor
                final_image = _to_pil(
                    context.ops.to_numpy(final_tensor), name="final output"
                )
            else:
                final_image = context.image.copy()
            guard_diagnostics: dict[str, object] = {
                "applied": False,
                "reason": "bypass_gate",
                "initial_ssim": None,
                "final_ssim": None,
                "alpha_before": 0.0,
                "alpha_after": 0.0,
                "trigger_ssim": 0.85,
                "target_ssim": 0.90,
            }
        elif context.ops is not None:
            final_tensor, guard_diagnostics = blend_with_ssim_guard(
                context.rgb_tensor,
                context.enhanced_tensor,
                alpha=alpha,
                input_median_luminance=float(context.scan_result.features.p50),
                ops=context.ops,
            )
            final_tensor = final_tensor.to(dtype=context.rgb_tensor.dtype)
            if tuple(final_tensor.shape) != tuple(context.rgb.shape):
                raise RuntimeError(
                    f"Final blend shape {tuple(final_tensor.shape)} != "
                    f"{context.rgb.shape}."
                )
            context.final_tensor = final_tensor
            final_image = _to_pil(
                context.ops.to_numpy(final_tensor), name="final output"
            )
        else:
            final_array, guard_diagnostics = blend_with_ssim_guard(
                context.rgb,
                context.enhanced,
                alpha=alpha,
                input_median_luminance=float(context.scan_result.features.p50),
            )
            if final_array.shape != context.rgb.shape:
                raise RuntimeError(
                    f"Final blend shape {final_array.shape} != {context.rgb.shape}."
                )
            final_image = _to_pil(final_array, name="final output")

        if context.capture_diagnostics:
            context.diagnostics["ssim_guard"] = guard_diagnostics
            context.diagnostics["effective_output_blend_alpha"] = float(
                guard_diagnostics["alpha_after"] or 0.0
            )

        if final_image.size != context.original_size:
            final_image.close()
            raise RuntimeError(
                f"Final image size {final_image.size} != {context.original_size}."
            )

        context.publish(ArtifactKind.FINAL_OUTPUT, final_image)
        return context


def default_stages() -> tuple[PipelineStage, ...]:
    """The ordered stage set used by the production pipeline."""
    return (
        AnalysisStage(),
        ClassicalStage(),
        DeepLearningStage(),
        FusionStage(),
        DenoisingStage(),
        ColourCorrectionStage(),
        QualityAssessmentStage(),
    )
