"""Tests for the section 5.8 final-output SSIM protection guard."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from skimage.metrics import structural_similarity

from backend.classical_cv.scan import scan
from backend.classical_cv.ssim_guard import blend_with_ssim_guard, ssim_rgb
from backend.enhancement.stages import QualityAssessmentStage, StageContext
from backend.enhancement.types import ArtifactKind


def _structured_source(size: int = 96) -> tuple[Image.Image, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    luminance = (
        0.34
        + 0.045 * np.sin(xx * 0.19)
        + 0.035 * np.cos(yy * 0.16)
        + 0.020 * np.sin((xx + yy) * 0.11)
    )
    rgb = np.repeat(np.clip(luminance, 0.0, 1.0)[..., None], 3, axis=2)
    image = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")
    source = np.asarray(image, dtype=np.float32) / 255.0
    return image, source


def _noisy_candidate(shape: tuple[int, int, int], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.20, 0.70, size=shape).astype(np.float32)


def _stage_context(
    image: Image.Image,
    source: np.ndarray,
    enhanced: np.ndarray,
    *,
    ops=None,
) -> StageContext:
    context = StageContext(
        image=image,
        mode="classical",
        original_size=image.size,
        capture_diagnostics=True,
    )
    context.rgb = source
    context.scan_result = scan(source)
    assert context.scan_result.features.p50 >= 0.25
    assert not context.scan_result.parameters.bypass_gate
    if ops is None:
        context.enhanced = enhanced
    else:
        context.ops = ops
        context.rgb_tensor = ops.ensure(source)
        context.enhanced_tensor = ops.ensure(enhanced)
    return context


def test_ssim_rgb_matches_evaluation_definition() -> None:
    rng = np.random.default_rng(22)
    source = rng.random((48, 53, 3), dtype=np.float32)
    output = np.clip(source * 0.72 + 0.12, 0.0, 1.0)

    actual = ssim_rgb(source, output)
    expected = structural_similarity(
        source, output, channel_axis=2, data_range=1.0
    )

    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("tensor_backend", [False, True])
def test_quality_stage_reduces_alpha_to_preserve_non_dark_structure(
    tensor_backend: bool,
) -> None:
    image, source = _structured_source()
    enhanced = _noisy_candidate(source.shape)
    ops = None
    if tensor_backend:
        pytest.importorskip("torch")
        from backend.runtime.config import resolve_runtime
        from backend.runtime.torch_ops import TorchImageOps

        ops = TorchImageOps(resolve_runtime("cpu"))
    context = _stage_context(image, source, enhanced, ops=ops)

    try:
        QualityAssessmentStage().run(context)

        output_image = context.outputs[ArtifactKind.FINAL_OUTPUT]
        output = np.asarray(output_image, dtype=np.float32) / 255.0
        diagnostics = context.diagnostics["ssim_guard"]

        assert diagnostics["applied"] is True
        assert diagnostics["initial_ssim"] < 0.85
        assert diagnostics["final_ssim"] >= 0.90
        assert diagnostics["alpha_after"] < diagnostics["alpha_before"]
        assert ssim_rgb(source, output) >= 0.90
        assert output_image.size == image.size
        if tensor_backend:
            assert context.final_tensor is not None
    finally:
        image.close()


def test_ssim_guard_skips_dark_inputs() -> None:
    source = np.full((32, 32, 3), 0.10, dtype=np.float32)
    enhanced = _noisy_candidate(source.shape)
    output, diagnostics = blend_with_ssim_guard(
        source, enhanced, alpha=0.8, input_median_luminance=0.10
    )

    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "dark_input"
    assert diagnostics["alpha_after"] == pytest.approx(0.8)
    assert np.isfinite(output).all()


def test_ssim_guard_does_not_change_a_similar_output() -> None:
    image, source = _structured_source()
    enhanced = np.clip(source * 1.02, 0.0, 1.0)
    output, diagnostics = blend_with_ssim_guard(
        source,
        enhanced,
        alpha=0.6,
        input_median_luminance=float(np.median(source[..., 0])),
    )

    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "initial_ssim_above_trigger"
    assert diagnostics["alpha_after"] == pytest.approx(0.6)
    assert ssim_rgb(source, output) >= 0.85
    image.close()
