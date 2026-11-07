# Training and offline tooling

This folder holds everything that runs offline: dataset preparation, the
metric harness entry point and (later) remote training notebooks. None of it is
imported by the FastAPI application.

## Layout

```
training/
├── data_pipeline/            # dataset preparation package
│   ├── config.py             # constants + exact split counts
│   ├── sources.py            # official sources and licences
│   ├── organisation.py       # discovery and low/high pairing
│   ├── validation.py         # corruption / truncation detection
│   ├── hashing.py            # 64-bit pHash + exact Hamming index
│   ├── leakage.py            # duplicate + cross-dataset leakage
│   ├── preprocessing.py      # decode, resize, crop, augment
│   ├── synthetic.py          # specified synth_dark formula
│   ├── splits.py             # deterministic split generation
│   ├── statistics.py         # cleaning reports
│   ├── pipeline.py           # orchestration
│   └── cli.py                # command line entry point
├── kaggle_eval/              # Kaggle discovery, runner, notebook, samples
├── benchmarks/               # reproducible LumenFuse backend benchmarks
├── experiments/              # stage attribution instrumentation/reporting
├── evaluate.py               # wrapper for backend.evaluation.cli
├── splits/                   # committed split spec / split files
└── reports/                  # committed cleaning reports
```

## Install

```bash
pip install -r requirements-eval.txt      # numpy, OpenCV, scikit-image, imagehash, SciPy, psutil
pip install -r requirements-metrics.txt   # optional: torch, torchvision, lpips, pyiqa
```

The API image only installs `requirements.txt`; the evaluation stack never
enters the deployed service.

## Dataset pipeline

```bash
python -m training.data_pipeline check --sources
python -m training.data_pipeline split --spec-only
python -m training.data_pipeline organise
python -m training.data_pipeline validate
python -m training.data_pipeline dedupe
python -m training.data_pipeline split
python -m training.data_pipeline preprocess --pool fine_tune --augment --darken
python -m training.data_pipeline synthetic --sources datasets/raw/sice/part1 --count 100
python -m training.data_pipeline report
python -m training.data_pipeline all
```

Raw images are read-only. Generated crops go to `datasets/processed/`; split
files and reports go to `training/splits/` and `training/reports/`.

Local data is present under `datasets/raw` (LOL-v1, LOL-v2, SICE_Paired and
extracted NPE/LIME/MEF/VV); ExDark/DICM are absent. See
[`../docs/datasets.md`](../docs/datasets.md) for the expected tree, the exact
counts, the sources to download manually and the cleaning rules.

## Evaluation

```bash
python -m training.evaluate --pairs pairs.json --output reports/eval
```

See [`../docs/evaluation.md`](../docs/evaluation.md) for the metric definitions,
aggregation and the integration-test split.

## LumenFuse backend benchmark

```bash
python -m training.benchmarks.benchmark_lumen_fuse \
  --device auto --warmups 2 --runs 10
```

On Kaggle T4, use `--device cuda`. The script compares the OpenCV CPU backend
with torch FP32/FP16 on identical candidates and reports cold/warm/median/mean/P90
latency, peak memory where available, utilization where measurable, and output
quality differences. See [`../docs/fusion_benchmarks.md`](../docs/fusion_benchmarks.md)
for measured local results and limitations; local MPS numbers are not T4 results.

## Full-pipeline GPU benchmark

`training.benchmarks.benchmark_full_pipeline_gpu` measures the complete
pipeline (all stages, all three modes) on NVIDIA CUDA only. It refuses
`cpu`/`mps`/`auto`, traps the CPU stage implementations so a silent fallback
fails the run, and writes JSON plus a Markdown report:

```bash
python -m training.benchmarks.benchmark_full_pipeline_gpu \
  --device cuda --precision float32 \
  --images datasets/raw/LOL-v2-Real/Test/Low --count 10 \
  --warmups 3 --runs 1 \
  --output-dir training/reports/full_pipeline_gpu
```

See [`../docs/gpu_pipeline.md`](../docs/gpu_pipeline.md) for the T4 command,
the reported statistics and the checks that must pass before a number is quoted.

## Kaggle evaluation

```bash
python -m training.kaggle_eval --discover-only \
  --input-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
  --additional-root /kaggle/input/datasets/washingtongold/exdark-dataset \
  --additional-root /kaggle/input/datasets/twozeb/dicm-img \
  --additional-root /kaggle/working \
  --output-dir /kaggle/working/lumenoir-evaluation
python -m training.kaggle_eval --config /kaggle/working/eval_config.json
```

The runner reuses `backend.enhancement` and `backend.evaluation`. It writes
deterministic manifests, image-level records, aggregates, latency, failure logs,
device information and before/after visuals under `/kaggle/working/`. LIME,
MEF, NPE/NPE-ex1/2/3 and VV are unpaired; NPE and its extensions are grouped
into one `npe` report with subgroup metadata. Both train and test records are
discovered, but the runner validates manifests and evaluates only test/eval
records. See [`kaggle_eval/README.md`](kaggle_eval/README.md) for the explicit
dataset configuration and limitations.

## Stage attribution experiment

```bash
python -m training.experiments.stage_attribution \
  --validation-manifest /path/to/validation.json \
  --data-root /path/to/datasets/raw \
  --output-dir /tmp/lumenoir-stage-attribution \
  --metrics psnr,ssim,lpips,delta_e00
```

This captures the actual original, raw Zero-DCE++, classical, fusion,
EdgeGuard, HueLock, final and ground-truth images and writes stage/dataset/
per-image/weight/latency reports. It refuses to run without an explicit
validation manifest; no validation split is created automatically. Manifest
paths are dataset-relative and resolved against `--data-root`, so the same
manifest can use `/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets`
on Kaggle. See [`experiments/README.md`](experiments/README.md).

## Fine-tuning (Zero-DCE++)

The reproducible fine-tuning entry point is `training.finetune`. It consumes
the existing `training/splits/train.json` and `validation.json` manifests,
keeps test images out, and writes fine-tuned checkpoints separately from the
released checkpoint:

```bash
LUMENOIR_EVAL_DEVICE=cuda python -m training.finetune \
  --config training/finetune/configs/zero_dce_plus_plus_lol.json \
  --output-dir /kaggle/working/lumenoir-zero-dce-finetune-$(date +%Y%m%d-%H%M%S)
```

The config controls CUDA AMP, batch size, resume state, scheduler, early
stopping and the documented loss terms. See [`finetune/README.md`](finetune/README.md).
Training is zero-reference on the low-light paired inputs; ground truth is used
only for validation PSNR/SSIM/LPIPS and checkpoint selection. No training is
started by the test suite.

## FP32 vs FP16 parity validation

`training.benchmarks.validate_fp16_parity_gpu` runs the same images through the
full pipeline in float32 and in float16 on one CUDA device and compares the
final images for all three modes: MAE, maximum absolute difference, PSNR and
SSIM, per image and aggregated, plus an `input | float32 | float16 | |diff| x8`
panel per image. It measures agreement between precisions, not enhancement
quality, and does not time anything.

```bash
python -m training.benchmarks.validate_fp16_parity_gpu \
  --device cuda \
  --images /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets --count 10 \
  --output-dir /kaggle/working/fp16_parity
```

It writes `fp32_vs_fp16_parity.json`, `fp32_vs_fp16_parity.md` and
`panels/<mode>/`. A report is only meaningful when the execution checks show
zero CPU stage calls and the audited parameter dtypes are `torch.float32` for
the reference run and `torch.float16` for the candidate.

This parity validator compares FP32 and FP16 numerical agreement, not enhancement quality. It requires an NVIDIA CUDA device and never silently falls back to CPU. It does not measure latency; use the full-pipeline benchmark for timing.
