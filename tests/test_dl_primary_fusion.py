"""Tests for the opt-in DL-primary detail-transfer fusion."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.fusion.dl_primary import (
    DLPrimaryFusionConfig,
    fuse_dl_primary,
)


def _detail_case(size: int = 64):
    yy, xx = np.mgrid[:size, :size]
    pattern = (0.28 + 0.035 * np.sin(xx * 0.85) + 0.025 * np.cos(yy * 0.65))
    original = np.repeat(pattern[..., None], 3, axis=2).astype(np.float32)
    classical = original.copy()
    deep = cv2.GaussianBlur(original, (0, 0), sigmaX=1.6)
    noise = np.zeros((size, size), dtype=np.float32)
    return original, classical, deep, noise


def test_dl_primary_preserves_dl_exactly_when_no_source_detail_evidence() -> None:
    original = np.full((32, 32, 3), 0.12, dtype=np.float32)
    classical = np.full_like(original, 0.40)
    deep = np.full_like(original, 0.32)
    noise = np.zeros((32, 32), dtype=np.float32)

    result = fuse_dl_primary(
        original, classical, deep, noise_map=noise
    )

    np.testing.assert_array_equal(result.image, deep)
    assert result.as_dict()["active_fraction"] == 0.0
    assert float(result.confidence.max()) == 0.0


def test_dl_primary_uses_classical_detail_only_when_source_supports_it() -> None:
    original, classical, deep, noise = _detail_case()

    result = fuse_dl_primary(
        original, classical, deep, noise_map=noise
    )

    assert np.isfinite(result.image).all()
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0
    assert float(result.confidence.mean()) > 0.0
    assert not np.array_equal(result.image, deep)
    assert (
        np.max(np.abs(result.correction))
        <= DLPrimaryFusionConfig().max_correction + 1e-7
    )


def test_dl_primary_disables_correction_in_high_noise_regions() -> None:
    original, classical, deep, noise = _detail_case()
    noise.fill(0.08)

    result = fuse_dl_primary(
        original, classical, deep, noise_map=noise
    )

    np.testing.assert_array_equal(result.image, deep)
    assert float(result.confidence.max()) == 0.0


def test_dl_primary_rejects_shape_mismatch_and_invalid_noise() -> None:
    original, classical, deep, noise = _detail_case()
    with pytest.raises(ValueError, match="identical"):
        fuse_dl_primary(original, classical[:-1], deep, noise_map=noise)
    noise[0, 0] = np.nan
    with pytest.raises(ValueError, match="noise_map"):
        fuse_dl_primary(original, classical, deep, noise_map=noise)


def test_dl_primary_torch_cpu_matches_numpy_backend() -> None:
    torch = pytest.importorskip("torch")
    from backend.runtime.torch_ops import TorchImageOps

    original, classical, deep, noise = _detail_case()
    expected = fuse_dl_primary(
        original, classical, deep, noise_map=noise
    ).image

    ops = TorchImageOps(device="cpu", precision="float32")
    actual = fuse_dl_primary(
        torch.from_numpy(original),
        torch.from_numpy(classical),
        torch.from_numpy(deep),
        noise_map=torch.from_numpy(noise),
        ops=ops,
    ).image

    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-6, rtol=1e-6)

def test_fusion_strategy_defaults_to_legacy_and_accepts_dl_primary(monkeypatch) -> None:
    from backend.enhancement.stages import _hybrid_fusion_strategy

    monkeypatch.delenv("LUMENOIR_HYBRID_STRATEGY", raising=False)
    assert _hybrid_fusion_strategy() == "legacy"

    monkeypatch.setenv("LUMENOIR_HYBRID_STRATEGY", "dl_primary")
    assert _hybrid_fusion_strategy() == "dl_primary"

    monkeypatch.setenv("LUMENOIR_HYBRID_STRATEGY", "unknown")
    with pytest.raises(ValueError, match="LUMENOIR_HYBRID_STRATEGY"):
        _hybrid_fusion_strategy()
