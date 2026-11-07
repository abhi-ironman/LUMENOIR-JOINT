"""Tests for the FP32 vs FP16 full-pipeline parity validation.

None of these tests need CUDA. They cover the metrics, the capture-before-
release rule, panel generation, report assembly and the CUDA-only gate. The
CUDA comparison itself only runs on an NVIDIA host, where the validator either
produces a report or refuses.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.benchmarks import validate_fp16_parity_gpu as parity


def _rgb(seed: int = 0, size: tuple[int, int] = (12, 10)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size[1], size[0], 3), dtype=np.float32)


def _write_image(path: Path, value: int = 40, size: tuple[int, int] = (12, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)
    return path


# ---------------------------------------------------------------------------
# CLI and CUDA gate
# ---------------------------------------------------------------------------


def test_parser_defaults_are_none() -> None:
    args = parity.build_parser().parse_args([])
    assert args.images is None
    assert args.device is None
    assert args.output_dir is None
    assert args.count is None
    assert args.modes is None
    assert args.weights is None


def test_settings_default_to_cuda_and_all_modes() -> None:
    settings = parity.settings_from_args(parity.build_parser().parse_args([]))
    assert settings.device == "cuda"
    assert settings.modes == parity.MODES
    assert settings.weights == parity.DEFAULT_WEIGHTS


def test_settings_reject_a_non_positive_count() -> None:
    args = parity.build_parser().parse_args(["--count", "0"])
    with pytest.raises(parity.ParityValidationError, match="positive integer"):
        parity.settings_from_args(args)


def test_settings_reject_an_unknown_mode() -> None:
    args = parity.build_parser().parse_args(["--modes", "fusion"])
    with pytest.raises(parity.gpu_bench.GpuBenchmarkError, match="Unknown mode"):
        parity.settings_from_args(args)


@pytest.mark.parametrize("device", ["cpu", "mps", "auto", ""])
def test_validation_refuses_non_cuda_devices(tmp_path: Path, device: str) -> None:
    _write_image(tmp_path / "images" / "a.png")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")
    args = parity.build_parser().parse_args(
        [
            "--images",
            str(tmp_path / "images"),
            "--weights",
            str(weights),
            "--device",
            device or "cpu",
        ]
    )
    with pytest.raises(parity.gpu_bench.GpuBenchmarkError):
        parity.run_validation(parity.settings_from_args(args))


def test_parity_error_is_a_benchmark_error() -> None:
    assert issubclass(parity.ParityValidationError, parity.gpu_bench.GpuBenchmarkError)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_identical_images_report_zero_difference() -> None:
    image = _rgb(1)
    metrics = parity.compare_images(image, image.copy())
    assert metrics["mae"] == 0.0
    assert metrics["max_abs"] == 0.0
    assert metrics["max_abs_8bit"] == 0.0
    assert metrics["identical"] is True
    assert metrics["psnr_db"] is None  # infinite dB is reported as not-a-number
    assert metrics["ssim"] == pytest.approx(1.0)


def test_known_constant_delta_produces_exact_mae_and_psnr() -> None:
    reference = _rgb(2)
    candidate = reference + 0.01
    metrics = parity.compare_images(reference, candidate)
    assert metrics["mae"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["max_abs"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["mae_8bit"] == pytest.approx(2.55, abs=1e-3)
    assert metrics["psnr_db"] == pytest.approx(40.0, abs=0.5)
    assert 0.9 < metrics["ssim"] < 1.0
    assert metrics["identical"] is False


def test_compare_images_rejects_shape_mismatch() -> None:
    with pytest.raises(parity.ParityValidationError, match="different shapes"):
        parity.compare_images(_rgb(3), _rgb(3, (13, 10)))


def test_compare_images_rejects_non_rgb_input() -> None:
    grey = np.zeros((10, 12), dtype=np.float32)
    with pytest.raises(parity.ParityValidationError, match="H, W, 3"):
        parity.compare_images(grey, grey)


def test_aggregate_metrics_summarises_per_image_values() -> None:
    entries = [
        {"mae": 0.001, "max_abs": 0.002, "psnr_db": 50.0, "ssim": 0.99, "identical": False},
        {"mae": 0.003, "max_abs": 0.004, "psnr_db": 46.0, "ssim": 0.97, "identical": True},
    ]
    summary = parity.aggregate_metrics(entries)
    assert summary["images"] == 2
    assert summary["identical_images"] == 1
    assert summary["mae_mean"] == pytest.approx(0.002)
    assert summary["mae_max"] == pytest.approx(0.003)
    assert summary["max_abs"] == pytest.approx(0.004)
    assert summary["psnr_mean_db"] == pytest.approx(48.0)
    assert summary["psnr_min_db"] == pytest.approx(46.0)
    assert summary["ssim_min"] == pytest.approx(0.97)


def test_aggregate_metrics_without_finite_psnr_returns_none() -> None:
    entries = [
        {"mae": 0.0, "max_abs": 0.0, "psnr_db": None, "ssim": 1.0, "identical": True}
    ]
    summary = parity.aggregate_metrics(entries)
    assert summary["psnr_mean_db"] is None
    assert summary["ssim_mean"] == pytest.approx(1.0)


def test_aggregate_metrics_rejects_an_empty_list() -> None:
    with pytest.raises(parity.ParityValidationError, match="empty"):
        parity.aggregate_metrics([])


# ---------------------------------------------------------------------------
# capture before release (requirement: never read a released buffer)
# ---------------------------------------------------------------------------


def test_capture_rgb_survives_closing_the_source_image() -> None:
    source = Image.new("RGB", (4, 3), (10, 20, 30))
    captured = parity.capture_rgb(source)
    source.close()

    assert captured.shape == (3, 4, 3)
    assert captured.dtype == np.float32
    assert np.allclose(captured[..., 0], 10 / 255.0)
    assert np.allclose(captured[..., 1], 20 / 255.0)
    assert np.allclose(captured[..., 2], 30 / 255.0)


class _FakeResult:
    """Mimics PipelineResult: release() invalidates the PIL handle."""

    def __init__(self, image: Image.Image) -> None:
        self.final_image: Image.Image | None = image
        self.status = type("Status", (), {"value": "completed"})()
        self.message = ""
        self.released = False

    def release(self) -> None:
        self.released = True
        if self.final_image is not None:
            self.final_image.close()
        self.final_image = None


class _FakePipeline:
    def __init__(self, value: int) -> None:
        self.value = value
        self.results: list[_FakeResult] = []

    def run(self, image: Image.Image, mode: str) -> _FakeResult:
        result = _FakeResult(Image.new("RGB", image.size, (self.value, 0, 0)))
        self.results.append(result)
        return result


class _FakeManager:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline
        self.model = object()
        self.status = None
        self.closed = False

    def load(self):
        self.status = type("Status", (), {"ready": True, "message": ""})()
        return self.status

    def close(self) -> None:
        self.closed = True


def test_collect_precision_outputs_copies_before_releasing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_image(tmp_path / "a.png", value=40)
    pipeline = _FakePipeline(value=200)
    manager = _FakeManager(pipeline)

    monkeypatch.setattr(parity, "ModelManager", lambda *_, **__: manager)
    monkeypatch.setattr(parity, "InferenceSettings", lambda *_, **kwargs: kwargs)
    monkeypatch.setattr(parity, "EnhancementPipeline", lambda *_, **__: pipeline)
    monkeypatch.setattr(
        parity.gpu_bench, "audit_model_device", lambda *_, **__: {"module_inspected": True}
    )
    monkeypatch.setattr(
        parity.gpu_bench, "audit_pipeline_ops", lambda *_, **__: {}
    )

    runtime = type("Runtime", (), {"precision": "float16", "device": "cuda:0"})()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")

    run = parity.collect_precision_outputs(
        runtime, [tmp_path / "a.png"], ("classical",), weights
    )

    captured = run.outputs["classical"]["a.png"]
    assert captured.shape == (10, 12, 3)
    assert captured[..., 0] == pytest.approx(200 / 255.0)
    assert pipeline.results[0].released is True
    assert manager.closed is True
    assert run.cpu_stage_calls == 0
    assert run.statuses["classical"]["a.png"] == "completed"


def test_collect_precision_outputs_raises_when_release_removes_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading after release must fail, not silently yield garbage."""

    class _ReleasingPipeline(_FakePipeline):
        def run(self, image: Image.Image, mode: str) -> _FakeResult:
            result = super().run(image, mode)
            result.release()
            return result

    _write_image(tmp_path / "a.png")
    pipeline = _ReleasingPipeline(value=200)
    manager = _FakeManager(pipeline)
    monkeypatch.setattr(parity, "ModelManager", lambda *_, **__: manager)
    monkeypatch.setattr(parity, "InferenceSettings", lambda *_, **kwargs: kwargs)
    monkeypatch.setattr(parity, "EnhancementPipeline", lambda *_, **__: pipeline)
    monkeypatch.setattr(
        parity.gpu_bench, "audit_model_device", lambda *_, **__: {"module_inspected": True}
    )
    monkeypatch.setattr(parity.gpu_bench, "audit_pipeline_ops", lambda *_, **__: {})

    runtime = type("Runtime", (), {"precision": "float16", "device": "cuda:0"})()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")

    with pytest.raises(parity.ParityValidationError, match="no final image"):
        parity.collect_precision_outputs(
            runtime, [tmp_path / "a.png"], ("classical",), weights
        )


# ---------------------------------------------------------------------------
# precision audit
# ---------------------------------------------------------------------------


def test_assert_model_precision_accepts_the_expected_dtype() -> None:
    parity.assert_model_precision(
        {"module_inspected": True, "parameter_dtypes": ["torch.float16"]}, "float16"
    )
    parity.assert_model_precision(
        {"module_inspected": True, "parameter_dtypes": ["torch.float32"]}, "float32"
    )


def test_assert_model_precision_rejects_a_wrong_dtype() -> None:
    with pytest.raises(parity.ParityValidationError, match=r"torch\.float32"):
        parity.assert_model_precision(
            {"module_inspected": True, "parameter_dtypes": ["torch.float32"]}, "float16"
        )


def test_assert_model_precision_rejects_an_uninspected_module() -> None:
    with pytest.raises(parity.ParityValidationError, match="Could not inspect"):
        parity.assert_model_precision({"module_inspected": False}, "float16")


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------


def test_panel_has_one_tile_per_image_and_labels_the_precisions() -> None:
    reference = _rgb(4)
    candidate = reference + 0.02
    panel = parity.build_panel(_rgb(5), reference, candidate, gain=8)
    try:
        assert panel.width == reference.shape[1] * 4
        assert panel.height == reference.shape[0]
    finally:
        panel.close()

    without_input = parity.build_panel(None, reference, candidate)
    try:
        assert without_input.width == reference.shape[1] * 3
    finally:
        without_input.close()


def test_difference_map_is_black_for_identical_images_and_bright_for_differences() -> None:
    reference = _rgb(6)
    assert parity.difference_map(reference, reference.copy()).max() == 0
    amplified = parity.difference_map(reference, reference + 0.05, gain=8)
    assert amplified.max() == round(0.05 * 8 * 255.0)
    assert amplified.shape == reference.shape
    saturating = parity.difference_map(reference, reference + 0.5, gain=8)
    assert saturating.max() == 255


def test_panel_path_groups_by_mode(tmp_path: Path) -> None:
    path = parity.panel_path(tmp_path / "panels", "hybrid", "image_01.png")
    assert path == tmp_path / "panels" / "hybrid" / "image_01__hybrid.png"


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------


def _runs(seed_offset: float = 0.0) -> tuple[parity.PrecisionRun, parity.PrecisionRun]:
    reference = parity.PrecisionRun(precision="float32", device="cuda:0")
    candidate = parity.PrecisionRun(precision="float16", device="cuda:0")
    for index, mode in enumerate(("classical", "deep_learning", "hybrid")):
        image = _rgb(10 + index)
        reference.outputs[mode] = {"a.png": image, "b.png": image * 0.5}
        candidate.outputs[mode] = {
            "a.png": image + seed_offset,
            "b.png": image * 0.5 + seed_offset,
        }
    return reference, candidate


def test_compare_runs_writes_panels_and_metrics_per_mode(tmp_path: Path) -> None:
    reference, candidate = _runs(0.01)
    inputs = {"a.png": _rgb(10), "b.png": _rgb(11)}
    sections = parity.compare_runs(
        reference,
        candidate,
        inputs,
        tmp_path / "panels",
        ("classical", "deep_learning", "hybrid"),
    )

    assert list(sections) == ["classical", "deep_learning", "hybrid"]
    for mode, section in sections.items():
        assert section["metrics"]["images"] == 2
        assert len(section["images"]) == 2
        for entry in section["images"]:
            for key in ("mae", "max_abs", "max_abs_8bit", "psnr_db", "ssim"):
                assert key in entry
            assert Path(entry["panel"]).is_file()
            assert Path(entry["panel"]).parent.name == mode


def test_compare_runs_rejects_a_missing_output(tmp_path: Path) -> None:
    reference, candidate = _runs()
    del candidate.outputs["hybrid"]["b.png"]
    with pytest.raises(parity.ParityValidationError, match="no float16 output"):
        parity.compare_runs(
            reference, candidate, {}, tmp_path / "panels", ("hybrid",)
        )


def _sample_report(tmp_path: Path) -> dict:
    reference, candidate = _runs(0.01)
    modes = parity.compare_runs(reference, candidate, {}, tmp_path / "panels", parity.MODES)
    return {
        "benchmark": "lumenoir.fp16_parity.gpu",
        "description": "float32 vs float16 agreement",
        "generated_at": "2026-11-07T00:00:00+00:00",
        "git_commit": "abc1234",
        "gpu": {
            "device": "cuda:0",
            "name": "Tesla T4",
            "compute_capability": "7.5",
            "total_memory_bytes": 16 * 1024**3,
            "total_memory_mb": 15360.0,
            "multi_processor_count": 40,
            "torch_version": "2.15.0",
            "cuda_version": "12.1",
            "cudnn_version": 8900,
            "python": "3.10.13",
            "platform": "Linux-5.15",
        },
        "settings": {
            "device": "cuda:0",
            "reference_precision": "float32",
            "candidate_precision": "float16",
            "weights": "backend/weights/zero_dce_plus_plus.pth",
            "image_source": "/kaggle/working/images",
            "count": 2,
            "modes": list(parity.MODES),
            "difference_gain": 8,
            "duplicate_inputs_per_precision": True,
        },
        "checks": {
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "same_device": True,
            "reference_model": {"parameter_dtypes": ["torch.float32"]},
            "candidate_model": {"parameter_dtypes": ["torch.float16"]},
            "cpu_stage_calls": {"float32": 0, "float16": 0},
            "outputs_captured_before_release": True,
        },
        "inputs": [{"name": "a.png", "path": "/kaggle/working/images/a.png", "sha256": "0" * 64}],
        "modes": modes,
    }


def test_render_markdown_covers_every_mode_and_metric(tmp_path: Path) -> None:
    text = parity.render_markdown(_sample_report(tmp_path))
    assert "Lumenoir FP32 vs FP16 full-pipeline parity" in text
    assert "Tesla T4" in text
    for mode in parity.MODES:
        assert f"## {mode}" in text
    for header in ("MAE", "Max abs", "PSNR mean dB", "SSIM mean"):
        assert header in text
    assert "| float32 parameter dtypes | ['torch.float32'] |" in text
    assert "| float16 parameter dtypes | ['torch.float16'] |" in text
    assert "Latency is not measured here" in text
    assert "not enhancement-quality" in text.replace("*", "")


def test_write_outputs_writes_valid_json_without_infinities(tmp_path: Path) -> None:
    report = _sample_report(tmp_path)
    json_path, markdown_path = parity.write_outputs(report, tmp_path / "out")

    assert json_path.name == "fp32_vs_fp16_parity.json"
    assert markdown_path.name == "fp32_vs_fp16_parity.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["panels"].endswith("panels")
    assert set(payload["modes"]) == set(parity.MODES)
    assert "Infinity" not in json_path.read_text(encoding="utf-8")
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Lumenoir FP32 vs FP16 full-pipeline parity"
    )


def test_psnr_of_identical_outputs_serialises_as_null(tmp_path: Path) -> None:
    reference, candidate = _runs(0.0)
    sections = parity.compare_runs(
        reference, candidate, {}, tmp_path / "panels", ("deep_learning",)
    )
    assert sections["deep_learning"]["metrics"]["identical_images"] == 2
    encoded = json.dumps(sections["deep_learning"], allow_nan=False)
    assert "Infinity" not in encoded
    assert sections["deep_learning"]["metrics"]["psnr_mean_db"] is None


def test_metrics_agree_with_the_reference_formula() -> None:
    """PSNR is 20*log10(1/rmse) for data_range = 1."""
    reference = _rgb(12)
    candidate = reference + 0.02
    metrics = parity.compare_images(reference, candidate)
    rmse = float(np.sqrt(np.mean((candidate - reference) ** 2)))
    assert metrics["psnr_db"] == pytest.approx(20.0 * math.log10(1.0 / rmse), abs=0.1)
