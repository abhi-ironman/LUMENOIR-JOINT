"""Fine-tuning config, data, loss and one-epoch trainer tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from backend.deep_learning.zero_dce_net import ZeroDCEPlusPlusNet
from backend.deep_learning.zero_dce_plus_plus import ZeroDCEPlusPlusModel
from training.finetune.config import (
    EarlyStoppingSettings,
    LossWeights,
    SchedulerSettings,
    TrainConfig,
)
from training.finetune.data import (
    PairRecord,
    PairedDataset,
    is_test_path,
    load_pairs,
    load_train_validation_pairs,
    make_loader,
)
from training.finetune.losses import zero_dce_loss
from training.finetune.metrics import validation_loss, validation_objective


def _image(path: Path, seed: int, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)


def _pair(root: Path, split: str, index: int) -> dict[str, str]:
    low = root / f"{split}/low/{index}.png"
    high = root / f"{split}/high/{index}.png"
    _image(low, index + 1)
    _image(high, index + 101)
    return {
        "dataset": "lol_v1",
        "image_id": f"{split}/low/{index}.png",
        "low": f"{split}/low/{index}.png",
        "high": f"{split}/high/{index}.png",
        "split": split,
        "kind": "paired",
    }


def _manifest(path: Path, records: list[dict[str, str]], pool: str) -> Path:
    path.write_text(
        json.dumps({"pool": pool, "records": records}),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, *, pretrained: Path, output: Path) -> TrainConfig:
    train_records = [_pair(tmp_path / "data", "train", index) for index in range(2)]
    val_records = [_pair(tmp_path / "data", "validation", index) for index in range(2)]
    train_manifest = _manifest(tmp_path / "train.json", train_records, "train")
    val_manifest = _manifest(tmp_path / "validation.json", val_records, "validation")
    return TrainConfig(
        train_manifest=train_manifest,
        validation_manifest=val_manifest,
        data_root=tmp_path / "data",
        pretrained_weights=pretrained,
        output_dir=output,
        device="cpu",
        amp=False,
        batch_size=2,
        num_workers=0,
        crop_size=16,
        epochs=1,
        log_every=1,
        save_every=1,
        loss=LossWeights(),
        scheduler=SchedulerSettings(name="reduce_on_plateau", patience=1),
        early_stopping=EarlyStoppingSettings(patience=1, min_delta=0.0),
    )



def _single_image(root: Path, split: str, index: int) -> dict[str, object]:
    low = root / "sice" / split / "single" / f"{index}.png"
    _image(low, index + 301)
    return {
        "id": f"sice:{split}/single/{index}.png",
        "dataset": "sice",
        "group": "part1",
        "low": f"sice/{split}/single/{index}.png",
        "high": None,
        "has_gt": False,
    }


def test_load_pairs_accepts_single_image_sice_from_full_split_pool(tmp_path) -> None:
    data_root = tmp_path / "data"
    record = _single_image(data_root, "part1", 0)
    manifest = _manifest(tmp_path / "fine_tune.json", [record], "fine_tune")

    loaded = load_pairs(
        manifest,
        expected_split="train",
        data_root=data_root,
    )

    assert len(loaded) == 1
    assert loaded[0].dataset == "sice"
    assert loaded[0].high is None
    assert loaded[0].low.is_file()


def test_unpaired_sice_is_rejected_from_test_and_do_no_harm_folders(tmp_path) -> None:
    data_root = tmp_path / "data"
    low = data_root / "sice/part2_lowlight/image.png"
    _image(low, 55)
    record = {
        "dataset": "sice",
        "image_id": "part2_lowlight/image.png",
        "low": "sice/part2_lowlight/image.png",
        "high": None,
        "has_gt": False,
    }
    manifest = _manifest(tmp_path / "train.json", [record], "fine_tune")

    with pytest.raises(ValueError, match="test/do-no-harm path"):
        load_pairs(manifest, expected_split="train", data_root=data_root)

    bright = data_root / "sice/do_no_harm/image.png"
    _image(bright, 56)
    assert is_test_path(bright, data_root)


def test_training_loader_batches_paired_and_unpaired_inputs(tmp_path) -> None:
    data_root = tmp_path / "data"
    _pair(data_root, "train", 0)
    single_record = _single_image(data_root, "part1", 0)
    paired = PairRecord(
        dataset="lol_v1",
        image_id="train/low/0.png",
        low=data_root / "train/low/0.png",
        high=data_root / "train/high/0.png",
    )
    single = PairRecord(
        dataset="sice",
        image_id=str(single_record["id"]),
        low=data_root / str(single_record["low"]),
        high=None,
    )

    dataset = PairedDataset(
        [paired, single],
        crop_size=16,
        train=True,
        seed=42,
        darkening_probability=0.0,
    )
    loader = make_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        seed=42,
        epoch=0,
    )
    batch = next(iter(loader))

    assert batch["low"].shape == (2, 3, 16, 16)
    assert batch["has_reference"] == [True, False]
    assert batch["dataset"] == ["lol_v1", "sice"]


def test_zero_reference_validation_loss_accepts_single_image_records(tmp_path) -> None:
    low = tmp_path / "sice/part1/image.png"
    _image(low, 57, size=32)
    record = PairRecord(dataset="sice", image_id="part1/image.png", low=low, high=None)
    net = ZeroDCEPlusPlusNet(scale_factor=1)

    loss = validation_loss(
        net,
        [record],
        weights=LossWeights(),
        device=torch.device("cpu"),
        amp=False,
    )

    assert np.isfinite(loss)


def test_config_round_trip_and_validation(tmp_path) -> None:
    pretrained = tmp_path / "pretrained.pth"
    torch.save(ZeroDCEPlusPlusNet(scale_factor=1).state_dict(), pretrained)
    config = _config(tmp_path, pretrained=pretrained, output=tmp_path / "out")
    config.validate()
    payload = config.as_dict()
    restored = TrainConfig.from_dict(payload, base=tmp_path)
    assert restored.output_dir == config.output_dir
    assert restored.loss == config.loss


def test_load_pairs_rejects_test_paths_and_overlap(tmp_path) -> None:
    test_record = _pair(tmp_path / "data", "eval15", 0)
    manifest = _manifest(tmp_path / "test.json", [test_record], "test")
    with pytest.raises(ValueError, match="test path"):
        load_pairs(manifest, expected_split="eval15", data_root=tmp_path / "data")

    train_record = _pair(tmp_path / "data", "train", 0)
    train_manifest = _manifest(tmp_path / "train.json", [train_record], "train")
    val_manifest = _manifest(
        tmp_path / "validation.json",
        [dict(train_record, split="validation")],
        "validation",
    )
    with pytest.raises(ValueError, match="leakage"):
        load_train_validation_pairs(train_manifest, val_manifest, tmp_path / "data")


def test_dataset_uses_same_crop_for_low_and_high(tmp_path) -> None:
    records = [_pair(tmp_path / "data", "train", index) for index in range(2)]
    pairs = load_pairs(
        _manifest(tmp_path / "train.json", records, "train"),
        expected_split="train",
        data_root=tmp_path / "data",
    )
    dataset = PairedDataset(pairs, crop_size=16, train=True, seed=42)
    item = dataset[0]
    assert item["low"].shape == (3, 16, 16)
    assert item["high"].shape == (3, 16, 16)
    second = dataset[0]
    assert torch.equal(item["low"], second["low"])
    assert torch.equal(item["high"], second["high"])


def test_documented_losses_are_finite_and_have_expected_terms() -> None:
    source = torch.rand(2, 3, 32, 32)
    output = (source * 1.2).clamp(0, 1)
    alpha = torch.zeros(2, 3, 32, 32)
    breakdown = zero_dce_loss(output, alpha, source, LossWeights())
    assert set(breakdown.terms) == {
        "spatial",
        "exposure",
        "color",
        "tv_alpha",
        "highlight",
        "hue",
    }
    assert torch.isfinite(breakdown.total)
    assert torch.isfinite(torch.stack(list(breakdown.terms.values()))).all()


def test_validation_objective_is_lower_better() -> None:
    better = validation_objective({"lpips": 0.2, "ssim": 0.8, "new_clipping": 0.0})
    worse = validation_objective({"lpips": 0.3, "ssim": 0.7, "new_clipping": 0.01})
    assert better < worse


def test_best_checkpoint_is_inference_compatible(tmp_path) -> None:
    net = ZeroDCEPlusPlusNet(scale_factor=1)
    checkpoint = tmp_path / "best.pth"
    torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, checkpoint)
    model = ZeroDCEPlusPlusModel(checkpoint, device="cpu", scale_factor=1)
    model.load()
    assert model.parameter_count == 10_561


def test_one_epoch_trainer_writes_checkpoints(tmp_path, monkeypatch) -> None:
    from training.finetune import trainer as trainer_module
    from training.finetune.trainer import FineTuner

    pretrained = tmp_path / "pretrained.pth"
    torch.save(ZeroDCEPlusPlusNet(scale_factor=1).state_dict(), pretrained)
    config = _config(tmp_path, pretrained=pretrained, output=tmp_path / "out")

    def fake_eval(net, pairs, *, stage, **kwargs):
        return [
            {
                "dataset": pair.dataset,
                "image_id": pair.image_id,
                "stage": stage,
                "psnr": 20.0,
                "ssim": 0.8,
                "lpips": 0.2,
                "new_clipping": 0.0,
            }
            for pair in pairs
        ]

    monkeypatch.setattr(trainer_module, "evaluate_network", fake_eval)
    monkeypatch.setattr(trainer_module, "validation_loss", lambda *args, **kwargs: 1.0)

    summary = FineTuner(config).fit()
    output = config.output_dir
    assert summary["epochs_completed"] == 1
    assert (output / "best.pth").is_file()
    assert (output / "last_training.pt").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "validation_comparison.csv").is_file()
    assert (output / "run_config.json").is_file()
    model = ZeroDCEPlusPlusModel(output / "best.pth", device="cpu", scale_factor=1)
    model.load()
    assert model.parameter_count == 10_561
