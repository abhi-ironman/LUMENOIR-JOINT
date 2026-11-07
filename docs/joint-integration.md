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

Record the fork commit SHA, environment variables, checkpoint filename and SHA-256, dataset manifest and hardware with every report. Use the same checkpoint path for all variants in a block. The currently available comparison is the published pretrained Zero-DCE++ checkpoint versus the project's own fine-tuned `best.pth`; a separate collaborator checkpoint is optional, not a prerequisite. Do not overwrite either source file.

### Recommended configuration matrix

1. **Fork baseline:** Retinex downsample `2`; source anchor `0`.
2. **Retinex sweep:** divisors `1`, `2`, `3`, `4`; anchor remains `0`.
3. **HueLock ablation:** chosen Retinex divisor; anchor `0` versus `0.45`.
4. **Combined candidate:** chosen Retinex divisor plus anchor `0.45`.

Use the joint checkpoint-matrix runner to automate the full comparison. It evaluates the Original/Classical reference once, then runs Deep Learning and Hybrid for every checkpoint and parameter combination. It fixes one Zero-DCE++ inference scale across all supplied checkpoints; the default is 1, matching the shared fine-tuning configuration.

```bash
python -m training.kaggle_eval.compare_checkpoints \
  --config training/kaggle_eval/example_config.json \
  --checkpoint pretrained=/absolute/path/to/zero_dce_plus_plus.pth \
  --checkpoint abhi=/absolute/path/to/best.pth \
  --datasets lol_v1,lol_v2_real,lol_v2_syn \
  --model-scale-factor 1 \
  --retinex-downsample 2 3 \
  --hue-anchor 0 0.45 \
  --device cuda \
  --output-dir training/reports/joint-matrix
```

The command creates a separate report directory for each checkpoint and parameter setting, plus `matrix_manifest.json` with checkpoint SHA-256 hashes, the selected model scale, experiment settings and run errors. Archive the exact Git commit SHA alongside the report. Use `--device cpu` when CUDA is unavailable. For a single checkpoint, supply one `--checkpoint` argument. Keep both original checkpoint files unchanged.

The runner writes per-image CSV/JSONL records, aggregates, failures and visual examples. Compare PSNR, SSIM, LPIPS and CIEDE2000 on paired sets; NIQE, noise, edge/texture, clipping, halo, black-crush and latency diagnostics on the same run.

Do not compare against previously reported aggregate means unless the images, preprocessing and metric implementations are confirmed identical. Do not use a test split for tuning and then label the same run as a blind final evaluation.

## Checkpoints and licensing

Model weights are not stored in Git. Keep the original checkpoint files separately, record their SHA-256 hashes, and supply the selected path through the evaluation configuration/`--weights` option or `LUMENOIR_MODEL_WEIGHTS`. The Zero-DCE++ implementation/checkpoint is documented in `docs/zero_dce.md` and `backend/weights/README.md`; its upstream CC BY-NC 4.0 restriction remains relevant. Review the licence and provenance of any separately fine-tuned checkpoint before redistribution or commercial use.

## Current verification status

The integration branch has passing push and pull-request CI runs, including the Phase Engine, SSIM guard and single-checkpoint comparison regression tests. A local Mac/MPS smoke test on 7 November 2026 confirmed that the model loads as `ready` and that the workspace returns Original, illumination/noise maps, enhancement-strength map, Classical, Deep Learning, Fusion and final-output artifacts. This was a single-image functional check, not a paired benchmark.

The §5.8 SSIM guard is implemented for CPU and tensor execution: on non-dark inputs (`p50 >= 0.25`), if the proposed input/output SSIM falls below `0.85`, the quality stage reduces the blend toward the source until the delivered output reaches at least `0.90`. The regression test checks both code paths.

A paired Kaggle evaluation is still required before promoting Retinex/HueLock settings or Phase Engine retry coefficients as measured optima. The current values remain experimental defaults.

An additional opt-in `dl_primary` hybrid strategy is available through
`LUMENOIR_HYBRID_STRATEGY=dl_primary` or the checkpoint comparison CLI flag.
It keeps Zero-DCE++ as the base and applies only a bounded, noise-gated
high-frequency luminance correction when the classical detail is more consistent
with the source detail. The legacy LumenFuse remains the default. The new
strategy is experimental and must be evaluated on held-out validation data
before it can be recommended or enabled by default.


## FP32 versus FP16 parity

The optional `training.benchmarks.validate_fp16_parity_gpu` tool runs the full pipeline on the same NVIDIA CUDA device in FP32 and FP16. It enforces CUDA, audits model parameter dtypes, traps CPU stage fallbacks and captures final images before pipeline results are released. It reports MAE, maximum absolute pixel difference, PSNR and SSIM between the two precision outputs, with visual difference panels.

These are *precision-agreement* metrics, not enhancement-quality metrics against ground truth. The tool does not measure latency; use `training.benchmarks.benchmark_full_pipeline_gpu` for performance. The Kaggle notebook provides an optional 10-image check for each supplied checkpoint on LOL-v2 Real.


## Optional closed-loop Phase Engine

The §5.7 Phase Engine is available on the integration branch. It runs one
baseline pass and at most two corrective retries, always starting from the
original input. The first applicable rule is chosen in the documented order:
reduce gain / lower the shoulder for new clipping; adjust the target brightness
when the input noise cap is not binding; strengthen EdgeGuard while reducing
gain for excess output noise; or reduce HueLock and DL-branch strength for
excess mean hue shift. It returns the completed pass with the lowest
Q = 10·new_clip + 2·|median_Y − 0.45| + 10·sigma_out.

Enable it with LUMENOIR_CLOSED_LOOP_ENABLED=true, or call
EnhancementPipeline.run(..., closed_loop=True). It is disabled by default,
so the existing baseline path and shared-stage cache are unchanged. Closed-loop
passes deliberately do not reuse cached prefix stages because their parameters
differ. The do-no-harm input bypass runs only the baseline pass.

The retry correction factors (0.80 gain after clipping, −0.10 shoulder start,
1.50 denoising multiplier, 0.85 gain after noise, and 0.80 HueLock/DL strength
after colour shift) are initial engineering settings, not measured optima.
Validate them on paired validation data before enabling this feature by default.
