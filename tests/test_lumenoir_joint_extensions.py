"""Regression tests for features ported from the standalone LUMENOIR branch."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.classical_cv.hue_lock import HueLockConfig, correct
from backend.classical_cv.retinex import LumenRetinexConfig, enhance
from backend.classical_cv.scan import scan


def _lab_distance(a: np.ndarray, b: np.ndarray) -> float:
    lab_a = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2LAB)
    lab_b = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.mean(np.sqrt(np.sum(np.square(lab_a - lab_b), axis=2))))


def test_source_chroma_anchor_is_disabled_by_default() -> None:
    source = np.full((64, 80, 3), 0.30, dtype=np.float32)
    enhanced = source.copy()
    enhanced[..., 2] = 0.42
    scan_result = scan(source)

    default = correct(enhanced, scan_result=scan_result)
    explicit_off = correct(
        enhanced,
        scan_result=scan_result,
        reference_rgb=source,
        config=HueLockConfig(source_anchor_strength=0.0),
    )

    assert np.array_equal(default.image, enhanced)
    assert np.array_equal(explicit_off.image, enhanced)
    assert default.source_anchor_mean_weight == 0.0


def test_source_chroma_anchor_reduces_cast_in_neutral_reference_regions() -> None:
    source = np.full((96, 128, 3), 0.30, dtype=np.float32)
    enhanced = source.copy()
    enhanced[..., 2] = 0.42
    scan_result = scan(source)

    baseline = correct(enhanced, scan_result=scan_result)
    anchored = correct(
        enhanced,
        scan_result=scan_result,
        reference_rgb=source,
        config=HueLockConfig(source_anchor_strength=0.45),
    )

    assert _lab_distance(source, anchored.image) < _lab_distance(source, baseline.image)
    assert anchored.source_anchor_mean_weight > 0.20
    assert anchored.source_anchor_coverage > 0.90
    assert anchored.image.shape == source.shape
    assert np.isfinite(anchored.image).all()
    assert anchored.image.min() >= 0.0
    assert anchored.image.max() <= 1.0


def test_source_chroma_anchor_protects_colourful_source_regions() -> None:
    source = np.full((64, 80, 3), (0.80, 0.10, 0.05), dtype=np.float32)
    enhanced = np.full_like(source, (0.25, 0.30, 0.48))
    scan_result = scan(source)

    baseline = correct(enhanced, scan_result=scan_result)
    anchored = correct(
        enhanced,
        scan_result=scan_result,
        reference_rgb=source,
        config=HueLockConfig(source_anchor_strength=0.45),
    )

    assert anchored.source_anchor_mean_weight < 1e-4
    assert np.mean(np.abs(anchored.image - baseline.image)) < 0.002


def test_source_chroma_anchor_does_not_trust_near_black_reference() -> None:
    source = np.full((64, 80, 3), 0.008, dtype=np.float32)
    enhanced = np.full_like(source, (0.10, 0.12, 0.20))
    anchored = correct(
        enhanced,
        scan_result=scan(source),
        reference_rgb=source,
        config=HueLockConfig(source_anchor_strength=0.45),
    )
    baseline = correct(enhanced, scan_result=scan(source))

    assert anchored.source_anchor_mean_weight < 1e-5
    assert np.mean(np.abs(anchored.image - baseline.image)) < 0.002


def test_retinex_supports_third_scale_illumination_estimation() -> None:
    rng = np.random.default_rng(64)
    image = np.clip(
        0.04 + rng.random((144, 192, 3), dtype=np.float32) * 0.18,
        0.0,
        1.0,
    )
    result = enhance(
        image,
        scan_result=scan(image),
        config=LumenRetinexConfig(illumination_downsample=3),
    )

    assert result.bypassed is False
    assert result.image.shape == image.shape
    assert result.illumination is not None
    assert result.illumination.shape == image.shape[:2]
    assert np.isfinite(result.image).all()
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0


def test_retinex_downsample_rejects_non_positive_factor() -> None:
    from backend.classical_cv.illumination import estimate_illumination

    image = np.full((48, 64, 3), 0.10, dtype=np.float32)
    with pytest.raises(ValueError, match="downsample"):
        estimate_illumination(image, downsample=0)


def test_pipeline_environment_controls_are_shared_and_validated(monkeypatch) -> None:
    from backend.enhancement.stages import (
        _hue_lock_config_from_environment,
        _retinex_config_from_environment,
    )

    monkeypatch.delenv("LUMENOIR_RETINEX_DOWNSAMPLE", raising=False)
    monkeypatch.delenv("LUMENOIR_HUELOCK_SOURCE_ANCHOR", raising=False)
    assert _retinex_config_from_environment().illumination_downsample == 2
    assert _hue_lock_config_from_environment().source_anchor_strength == 0.0

    monkeypatch.setenv("LUMENOIR_RETINEX_DOWNSAMPLE", "3")
    monkeypatch.setenv("LUMENOIR_HUELOCK_SOURCE_ANCHOR", "0.45")
    assert _retinex_config_from_environment().illumination_downsample == 3
    assert _hue_lock_config_from_environment().source_anchor_strength == 0.45

    monkeypatch.setenv("LUMENOIR_RETINEX_DOWNSAMPLE", "0")
    with pytest.raises(ValueError, match="LUMENOIR_RETINEX_DOWNSAMPLE"):
        _retinex_config_from_environment()
    monkeypatch.setenv("LUMENOIR_HUELOCK_SOURCE_ANCHOR", "0.9")
    with pytest.raises(ValueError, match="LUMENOIR_HUELOCK_SOURCE_ANCHOR"):
        _hue_lock_config_from_environment()


def test_tensor_hue_lock_supports_source_chroma_anchor() -> None:
    torch = pytest.importorskip("torch")
    from backend.classical_cv.torch_hue_lock import correct_tensor
    from backend.runtime.torch_ops import TorchImageOps

    source = np.full((48, 64, 3), 0.30, dtype=np.float32)
    enhanced = source.copy()
    enhanced[..., 2] = 0.42
    ops = TorchImageOps(device="cpu", precision="float32")
    source_tensor = torch.from_numpy(source)
    enhanced_tensor = torch.from_numpy(enhanced)

    result = correct_tensor(
        enhanced_tensor,
        scan_result=scan(source),
        reference_rgb=source_tensor,
        config=HueLockConfig(source_anchor_strength=0.45),
        ops=ops,
    )
    output = ops.to_numpy(result.image)

    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert _lab_distance(source, output) < _lab_distance(source, enhanced)
    assert result.source_anchor_mean_weight > 0.20
