"""Controlled checkpoint-matrix runner tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from training.kaggle_eval import compare_checkpoints


def test_matrix_accepts_one_pretrained_checkpoint_and_records_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "zero_dce_plus_plus.pth"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    config_path = tmp_path / "eval_config.json"
    config_path.write_text(
        json.dumps(
            {
                "input_root": str(tmp_path / "datasets"),
                "additional_roots": [],
                "selected_datasets": ["lol_v1"],
                "modes": ["hybrid"],
                "device": "cpu",
                "latency_warmups": 0,
                "latency_runs": 0,
                "skip_metrics": ["lpips", "niqe"],
            }
        ),
        encoding="utf-8",
    )

    reference_calls: list[object] = []
    model_calls: list[dict[str, object]] = []

    def fake_run_evaluation(config):
        reference_calls.append(config)
        return SimpleNamespace(
            output_dir=config.output_dir,
            record_count=15,
            failure_count=0,
            skipped_modes=(),
            warnings=(),
            aggregate_paths=(config.output_dir / "aggregate.csv",),
        )

    def fake_run_one(
        config,
        output_dir,
        *,
        retinex_downsample,
        hue_anchor,
        model_scale_factor,
        hybrid_strategy,
    ):
        assert config.weights == checkpoint.resolve()
        model_calls.append(
            {
                "output_dir": output_dir,
                "retinex_downsample": retinex_downsample,
                "hue_anchor": hue_anchor,
                "model_scale_factor": model_scale_factor,
                "hybrid_strategy": hybrid_strategy,
                "modes": config.modes,
            }
        )
        return {
            "retinex_downsample": retinex_downsample,
            "hue_lock_source_anchor": hue_anchor,
            "model_scale_factor": model_scale_factor,
            "output_dir": str(output_dir),
            "record_count": 15,
            "failure_count": 0,
            "skipped_modes": [],
            "warnings": [],
            "aggregate_paths": [],
        }

    monkeypatch.setattr(compare_checkpoints, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(compare_checkpoints, "_run_one", fake_run_one)

    output_dir = tmp_path / "matrix"
    exit_code = compare_checkpoints.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            f"pretrained={checkpoint}",
            "--datasets",
            "lol_v1",
            "--device",
            "cpu",
            "--latency-warmups",
            "0",
            "--latency-runs",
            "0",
            "--model-scale-factor",
            "1",
            "--hybrid-strategy",
            "dl_primary",
            "--retinex-downsample",
            "2",
            "3",
            "--hue-anchor",
            "0",
            "0.45",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads(
        (output_dir / "matrix_manifest.json").read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert len(reference_calls) == 1
    assert reference_calls[0].modes == ("original", "classical")
    assert reference_calls[0].weights is None
    assert len(model_calls) == 4
    assert all(call["modes"] == ("deep_learning", "hybrid") for call in model_calls)
    assert all(call["hybrid_strategy"] == "dl_primary" for call in model_calls)
    assert manifest["hybrid_strategy"] == "dl_primary"
    assert manifest["checkpoints"] == [
        {
            "label": "pretrained",
            "path": str(checkpoint.resolve()),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    ]
    assert manifest["reference_run"]["record_count"] == 15
    assert len(manifest["model_runs"]) == 4
    assert manifest["errors"] == []
