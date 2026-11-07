# Zero-DCE++ fine-tuning

This package fine-tunes the existing Zero-DCE++ curve network on the paired
LOL training split. It does not change the inference architecture, the
LumenFuse/LumenScan post-processing, the released checkpoint, or the evaluation
metrics.

## Inputs and leakage

- Training manifest: `training/splits/train.json` (1,565 pairs).
- Validation manifest: `training/splits/validation.json` (391 pairs).
- Test images are never loaded by the trainer. The split builder already
  excluded 118 train/validation candidates that matched test images, and the
  trainer rejects any path containing a test split component.
- `train` and `validation` path sets must be disjoint; training aborts if they
  overlap.
- The official checkpoint is read-only input. It is never copied over or
  overwritten. Fine-tuned outputs go to `--output-dir`:
  - `best.pth`: bare `state_dict`, accepted by the existing strict inference
    wrapper.
  - `last_training.pt`: resumable optimizer/scheduler/scaler/epoch state.
  - `train.log`, `history.csv`, `history.json`, `validation_metrics.csv`,
    `baseline_metrics.csv`, `validation_comparison.csv`, `run_config.json`,
    `summary.md`, `summary.json`.

### Current materialized manifest scope

The committed `training/splits/train.json` (1,565 records) and
`validation.json` (391 records) currently contain LOL-v1, LOL-v2 Real and
LOL-v2 Synthetic only. Although `split_spec.json` declares SICE Part 1
(2,422 training images and 600 validation images), those SICE records are not
present in the checked-in manifests. The current fine-tuning loader requires
paired low/high records and does not yet consume SICE Part 1's single-image
zero-reference records.

Therefore, the repository's default `zero_dce_plus_plus_lol.json` run is an
LOL-only fine-tune, not the full SICE+LOL recipe in `final_decision.md` §1.3.
Do not report SICE as part of the training data unless the actual
`run_config.json` and manifests for that run confirm it. Extending the
trainer to mix single-image SICE records with paired LOL validation data is a
separate implementation task; it should be completed before claiming the full
planned recipe.

## Loss terms

The defaults are the documented Zero-DCE++ recipe plus the two LUMENOIR terms
from `final_decision.md` §2.6:

| Term | Purpose | Default |
| --- | --- | ---: |
| `L_spa` | Preserve local contrast between 4×4 pooled regions | 1.0 |
| `L_exp` | Pull 16×16 local mean luminance toward 0.6 | 10.0 |
| `L_col` | Channel-mean colour constancy, reduced 30% from 5.0 | 3.5 |
| `L_tvA` | Smooth the predicted curve map | 1600.0 |
| `L_hl` | Penalise brightening already-bright pixels via a smoothstep mask | 5.0 |
| `L_hue` | Preserve RGB direction, preventing gray-world colour loss | 2.0 |

`L_spa`, `L_exp`, `L_col` and `L_tvA` are the Zero-DCE definitions. `L_hl` and
`L_hue` are the documented LUMENOIR additions; they are training objectives,
not inference architecture. No perceptual loss or supervised L1 objective is
enabled by default.

The validation objective is lower-is-better:

```text
V = LPIPS + 0.5 * (1 - SSIM) + 5 * max(0, new_clipping)
```

PSNR, SSIM and LPIPS are computed against the same ground truth for both the
released checkpoint and the current fine-tuned checkpoint.

## Reproducibility

- Seed 42 is applied to Python, NumPy, PyTorch and CUDA.
- cuDNN deterministic mode is enabled; deterministic data crops use an
  epoch/index-derived NumPy generator.
- AMP is enabled only on CUDA. Mixed-precision kernels can still vary slightly
  across GPU models; the config and seeds are written to `run_config.json`.

## Kaggle setup

Attach the repository code and the permanent dataset. Place the official
checkpoint in `/kaggle/working` without modifying it:

```bash
cd /kaggle/working/LUMENOIR
pip install -r requirements-training.txt

# The split manifests use dataset-relative paths. The config points at the
# permanent mount; verify the three LOL directory names exist below it.
ls /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets
```

Dry-run first (validates manifests and path resolution without training):

```bash
LUMENOIR_EVAL_DEVICE=cuda python -m training.finetune \
  --config training/finetune/configs/zero_dce_plus_plus_lol.json \
  --output-dir /kaggle/working/lumenoir-zero-dce-finetune-$(date +%Y%m%d-%H%M%S) \
  --dry-run
```

Start training only after explicit approval:

```bash
LUMENOIR_EVAL_DEVICE=cuda python -m training.finetune \
  --config training/finetune/configs/zero_dce_plus_plus_lol.json \
  --output-dir /kaggle/working/lumenoir-zero-dce-finetune-$(date +%Y%m%d-%H%M%S)
```

Resume from the last completed epoch:

```bash
LUMENOIR_EVAL_DEVICE=cuda python -m training.finetune \
  --config training/finetune/configs/zero_dce_plus_plus_lol.json \
  --output-dir /kaggle/working/lumenoir-zero-dce-finetune-20260101-120000 \
  --resume /kaggle/working/lumenoir-zero-dce-finetune-20260101-120000/last_training.pt
```

The command refuses to use a non-empty output directory unless resuming.

## Configurable settings

`training/finetune/configs/zero_dce_plus_plus_lol.json` exposes batch size,
workers, crop size, epochs, Adam learning rate/weight decay, gradient clipping,
AMP, scheduler, early stopping, loss weights, validation limits and resume
path. The config and resolved device are copied to `run_config.json`.

`scale_factor` is 1 during training, as specified in `final_decision.md`.
The checkpoint contains no scale-dependent parameters; deployment can still use
the existing inference default.

## Verification

```bash
python -m pytest -q tests/test_finetune.py
python -m pytest -q
python -m ruff check backend tests training
```

The one-epoch trainer test uses a randomly initialized architecture checkpoint
and stubbed validation metrics; it does not train on repository datasets or
download LPIPS weights.

## Limitations

- This is fine-tuning of the existing 10,561-parameter network, not a
  reimplementation or architecture change.
- Test images are not used for training, validation, checkpoint selection or
  tuning.
- The released checkpoint remains at `backend/weights/zero_dce_plus_plus.pth`
  (or the Kaggle input path); fine-tuned checkpoints are separate.
