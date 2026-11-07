# Joint LUMENOIR integration and comparison protocol

## Repository boundaries

This repository is the user-owned GitHub fork of `Chiranthcommits25/Lumenoir`. The upstream repository is treated as read-only. Integration work is developed on `integration/lumenoir-research` and is not merged into this fork's `main` until reviewed and tested.

The original standalone implementation remains at `abhi-ironman/LUMENOIR`. Its source and benchmark results are retained for provenance; its measurements are not automatically attributed to this fork.

## What this branch adds

The existing FastAPI/React application, CPU reference path, CUDA/MPS tensor path, stage cache, Zero-DCE++ model manager, dataset preparation and Kaggle evaluation runner remain the foundation.

Two experimental controls are wired through the same pipeline stage classes for CPU and tensor/GPU execution:

| Environment variable | Default | Allowed values | Purpose |
|---|---:|---|---|
| `LUMENOIR_RETINEX_DOWNSAMPLE` | `2` | integer 1–4 | Working-resolution divisor used to estimate the smooth Retinex illumination map. |
| `LUMENOIR_HUELOCK_SOURCE_ANCHOR` | `0` | float 0–0.55 | Optional confidence-weighted source-chroma anchor in HueLock. |

The baseline defaults deliberately preserve the fork's prior behaviour. The source anchor is off unless explicitly enabled. It trusts a source pixel only when it is sufficiently illuminated and low-chroma; the confidence is reduced as measured noise rises. The full-resolution final Lab transform is retained, while the colour-confidence maps and HueLock spread analysis are computed on a reduced analysis view (maximum side 320 px).

The Retinex downsample divisor here is **not** the 0.33 multiplier previously tested in the separate standalone LUMENOIR codebase. This fork uses a LIME-style guided-filter illumination estimate where divisor 2 was already the default. The 0.33 result from the other implementation is not assumed to transfer.

## Strict A/B experiment

For a valid comparison, hold constant:

- test manifest and exact relative image paths;
- model architecture, checkpoint file and Zero-DCE++ scale factor;
- image decoding, EXIF handling, resize policy and colour space;
- device, precision, OpenCV/PyTorch thread counts;
- warm-up and timed-run counts;
- installed metric packages and metric configuration.

Record the fork commit SHA, environment variables, checkpoint filename and SHA-256, dataset manifest and hardware with every report. Use the same checkpoint path for all variants in a block. To compare the two contributors' checkpoints, run the complete matrix once per checkpoint; do not overwrite the only copy of either checkpoint.

### Recommended configuration matrix

1. **Fork baseline:** Retinex downsample `2`; source anchor `0`.
2. **Retinex sweep:** divisors `1`, `2`, `3`, `4`; anchor remains `0`.
3. **HueLock ablation:** chosen Retinex divisor; anchor `0` versus `0.45`.
4. **Combined candidate:** chosen Retinex divisor plus anchor `0.45`.

Run the existing Kaggle evaluation runner for each configuration. Example (edit dataset roots and checkpoint path for the active environment):

```bash
export LUMENOIR_RETINEX_DOWNSAMPLE=2
export LUMENOIR_HUELOCK_SOURCE_ANCHOR=0
python -m training.kaggle_eval \\
  --config training/kaggle_eval/example_config.json \\
  --modes original,classical,deep_learning,hybrid \\
  --weights /absolute/path/to/checkpoint.pth \\
  --device cuda \\
  --output-dir training/reports/joint/baseline
```

Repeat with separate output directories for each matrix row. Use `--device cpu` where CUDA is unavailable. The runner writes per-image CSV/JSONL records, aggregates, failures and visual examples. Compare PSNR, SSIM, LPIPS and CIEDE2000 on paired sets; NIQE, noise, edge/texture, clipping, halo, black-crush and latency diagnostics on the same run.

Do not compare against previously reported aggregate means unless the images, preprocessing and metric implementations are confirmed identical. Do not use a test split for tuning and then label the same run as a blind final evaluation.

## Checkpoints and licensing

Model weights are not stored in Git. Keep the original checkpoint files separately, record their SHA-256 hashes, and supply the selected path through the evaluation configuration/`--weights` option or `LUMENOIR_MODEL_WEIGHTS`. The Zero-DCE++ implementation/checkpoint is documented in `docs/zero_dce.md` and `backend/weights/README.md`; its upstream CC BY-NC 4.0 restriction remains relevant. Review the licence and provenance of any separately fine-tuned checkpoint before redistribution or commercial use.

## Current verification status

The new CPU/GPU HueLock paths, stage environment wiring and regression tests have been committed to this integration branch. A clean CI run and a paired Kaggle evaluation are required before enabling source anchoring by default or changing the Retinex divisor.
