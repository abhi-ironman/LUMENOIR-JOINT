"""Unit tests for the bounded closed-loop Phase Engine."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from backend.classical_cv.scan import scan
from backend.enhancement.phase_engine import (
    PhaseEngine,
    PhaseEngineConfig,
    PhaseOverrides,
    apply_phase_overrides,
    input_noise_cap_is_binding,
    measure_phase_metrics,
)
from backend.enhancement.pipeline import EnhancementPipeline
from backend.enhancement.types import PipelineResult, ProcessingStatus


def _result(array: np.ndarray, total_ms: float = 2.0) -> PipelineResult:
    pixels = np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    return PipelineResult(
        status=ProcessingStatus.COMPLETED,
        mode="classical",
        stages=(),
        artifacts=(),
        total_ms=total_ms,
        final_image=image,
    )


def test_proxy_metrics_are_zero_for_a_neutral_flat_output() -> None:
    source = np.full((64, 64, 3), 0.20, dtype=np.float32)
    output = np.full((64, 64, 3), 0.45, dtype=np.float32)

    metrics = measure_phase_metrics(source, output)

    assert metrics.new_clipped_fraction == pytest.approx(0.0)
    assert metrics.median_luminance == pytest.approx(0.45, abs=0.01)
    assert metrics.noise_sigma == pytest.approx(0.0, abs=1e-8)
    assert metrics.mean_hue_shift_degrees == pytest.approx(0.0)
    assert metrics.proxy_score == pytest.approx(0.0, abs=0.02)


def test_new_clipping_counts_pixels_not_clipped_in_the_source() -> None:
    source = np.zeros((10, 10, 3), dtype=np.float32)
    output = np.zeros_like(source)
    source.reshape(-1, 3)[:20] = 1.0
    output.reshape(-1, 3)[20:40] = 1.0

    metrics = measure_phase_metrics(source, output)

    # Equal input/output clipped percentages do not imply zero newly clipped pixels.
    assert metrics.new_clipped_fraction == pytest.approx(0.20)


def test_clipping_rule_has_priority_and_best_pass_is_returned() -> None:
    source = np.full((64, 64, 3), 0.20, dtype=np.float32)
    clipped = np.ones_like(source)
    corrected = np.full_like(source, 0.45)
    seen: list[PhaseOverrides] = []

    def run_pass(overrides: PhaseOverrides) -> PipelineResult:
        seen.append(overrides)
        return _result(clipped if len(seen) == 1 else corrected)

    result = PhaseEngine().run(original=source, run_pass=run_pass)

    try:
        assert len(seen) == 2
        assert seen[0] == PhaseOverrides()
        assert seen[1].gain_cap_scale == pytest.approx(0.80)
        assert seen[1].shoulder_start_delta == pytest.approx(-0.10)
        assert np.asarray(result.final_image).mean() == pytest.approx(115, abs=1)
        engine_report = result.diagnostics["phase_engine"]
        assert engine_report["selected_pass"] == 1
        assert engine_report["attempt_count"] == 2
        assert engine_report["stop_reason"] == "proxy_thresholds_met"
    finally:
        result.release()


def test_phase_engine_stops_when_proxy_score_does_not_improve() -> None:
    source = np.full((64, 64, 3), 0.20, dtype=np.float32)
    rng = np.random.default_rng(4)
    noisy = np.clip(
        0.40 + rng.normal(0.0, 0.05, (64, 64, 3)), 0.0, 1.0
    ).astype(np.float32)
    seen: list[PhaseOverrides] = []

    def run_pass(overrides: PhaseOverrides) -> PipelineResult:
        seen.append(overrides)
        return _result(noisy)

    result = PhaseEngine().run(original=source, run_pass=run_pass)

    try:
        assert len(seen) == 2
        assert seen[1].denoise_strength_scale == pytest.approx(1.5)
        assert seen[1].gain_cap_scale == pytest.approx(0.85)
        assert result.diagnostics["phase_engine"]["selected_pass"] == 0
        assert result.diagnostics["phase_engine"]["stop_reason"] == (
            "proxy_score_did_not_improve"
        )
    finally:
        result.release()


def test_phase_engine_never_runs_more_than_two_extra_passes() -> None:
    source = np.full((64, 64, 3), 0.10, dtype=np.float32)
    outputs = [0.90, 0.80, 0.70]
    calls = 0

    def run_pass(_overrides: PhaseOverrides) -> PipelineResult:
        nonlocal calls
        value = outputs[calls]
        calls += 1
        return _result(np.full_like(source, value))

    result = PhaseEngine().run(original=source, run_pass=run_pass)

    try:
        assert calls == 3
        assert result.diagnostics["phase_engine"]["extra_passes"] == 2
        assert result.diagnostics["phase_engine"]["stop_reason"] == "max_extra_passes"
        assert np.asarray(result.final_image).mean() == pytest.approx(179, abs=1)
    finally:
        result.release()


def test_hue_shift_reduces_both_hue_lock_and_dl_strength() -> None:
    source = np.zeros((64, 64, 3), dtype=np.float32)
    source[..., 0] = 1.0
    output = np.zeros_like(source)
    output[..., 1] = 1.0

    metrics = measure_phase_metrics(source, output)
    assert metrics.mean_hue_shift_degrees == pytest.approx(120.0, abs=1.0)

    source = np.tile(np.array([0.80, 0.10, 0.10], dtype=np.float32), (64, 64, 1))
    output = np.tile(np.array([0.10, 0.80, 0.10], dtype=np.float32), (64, 64, 1))
    seen: list[PhaseOverrides] = []

    def run_pass(overrides: PhaseOverrides) -> PipelineResult:
        seen.append(overrides)
        return _result(output)

    result = PhaseEngine().run(original=source, run_pass=run_pass)
    try:
        assert len(seen) == 2
        assert seen[1].hue_beta_scale == pytest.approx(0.80)
        assert seen[1].dl_strength_scale == pytest.approx(0.80)
    finally:
        result.release()


def test_phase_overrides_adjust_scan_parameters_without_mutating_baseline() -> None:
    source = np.full((64, 64, 3), 0.15, dtype=np.float32)
    scan_result = scan(source)
    baseline = scan_result.parameters
    overrides = PhaseOverrides(
        gain_cap_scale=0.8,
        target_brightness=0.50,
        hue_beta_scale=0.75,
    )

    adjusted = apply_phase_overrides(scan_result, overrides)

    assert adjusted.parameters.gain_cap == pytest.approx(
        max(1.0, baseline.gain_cap * 0.8)
    )
    assert adjusted.parameters.gamma != baseline.gamma
    assert adjusted.parameters.hue_lock_beta == pytest.approx(
        baseline.hue_lock_beta * 0.75
    )
    assert scan_result.parameters == baseline


def test_do_no_harm_bypass_runs_only_the_baseline_pass() -> None:
    source = np.tile(
        np.linspace(0.20, 0.85, 64, dtype=np.float32)[None, :, None],
        (64, 1, 3),
    )
    calls = 0

    def run_pass(_overrides: PhaseOverrides) -> PipelineResult:
        nonlocal calls
        calls += 1
        return _result(source)

    result = PhaseEngine().run(original=source, run_pass=run_pass)

    try:
        assert calls == 1
        assert result.diagnostics["phase_engine"]["stop_reason"] == (
            "do_no_harm_bypass"
        )
    finally:
        result.release()


def test_config_rejects_more_than_two_extra_passes() -> None:
    with pytest.raises(ValueError, match="between 0 and 2"):
        PhaseEngineConfig(max_extra_passes=3)



def test_brightness_correction_respects_the_input_noise_cap() -> None:
    source = np.full((64, 64, 3), 0.10, dtype=np.float32)
    bright = np.full_like(source, 0.80)
    seen: list[PhaseOverrides] = []

    def run_pass(overrides: PhaseOverrides) -> PipelineResult:
        seen.append(overrides)
        return _result(bright)

    result = PhaseEngine().run(original=source, run_pass=run_pass)
    try:
        assert len(seen) == 2
        assert seen[1].target_brightness == pytest.approx(0.38)
    finally:
        result.release()

    rng = np.random.default_rng(42)
    noisy_gray = np.clip(
        0.30 + rng.normal(0.0, 0.05, (64, 64)), 0.0, 1.0
    ).astype(np.float32)
    noisy_source = np.repeat(noisy_gray[..., None], 3, axis=2)
    assert input_noise_cap_is_binding(noisy_source)

    calls = 0

    def run_noisy(_overrides: PhaseOverrides) -> PipelineResult:
        nonlocal calls
        calls += 1
        return _result(bright)

    capped = PhaseEngine().run(original=noisy_source, run_pass=run_noisy)
    try:
        assert calls == 1
        assert capped.diagnostics["phase_engine"]["noise_cap_binding"] is True
        assert capped.diagnostics["phase_engine"]["stop_reason"] == (
            "no_matching_correction"
        )
    finally:
        capped.release()


def test_pipeline_phase_engine_is_opt_in() -> None:
    image = Image.new("RGB", (64, 64), (15, 15, 15))
    pipeline = EnhancementPipeline()

    baseline = pipeline.run(image, mode="classical", closed_loop=False)
    enabled = pipeline.run(image, mode="classical", closed_loop=True)

    try:
        assert "phase_engine" not in baseline.diagnostics
        assert "phase_engine" in enabled.diagnostics
        assert 1 <= enabled.diagnostics["phase_engine"]["attempt_count"] <= 3
    finally:
        baseline.release()
        enabled.release()
        image.close()
