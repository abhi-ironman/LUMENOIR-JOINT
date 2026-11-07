# Evaluation framework

The full enhancement pipeline and the metrics harness are implemented. **No
standardized benchmark numbers are reported yet** because the paired and
unpaired research datasets have not been run through the final integration
branch. A successful local workspace upload is a smoke test, not a dataset
benchmark; the framework ensures the eventual comparison uses the same images,
metrics and reporting rules.

Definitions follow `final_decision.md` §6.

## Conventions

- Arrays are float32 RGB in [0, 1]; inputs are validated once in
  `backend/evaluation/images.py`.
- Reference-based metrics require a ground-truth image and are registered
  separately from no-reference metrics.
- A metric that cannot run (missing dependency, missing reference, invalid
  input) returns `value=None` with a note. Nothing is fabricated.

## Metrics

| Key | Label | Reference | Definition |
| --- | --- | --- | --- |
| `psnr` | PSNR | yes | `skimage.metrics.peak_signal_noise_ratio`, `data_range=1.0` |
| `psnr_aligned` | PSNR (mean-aligned) | yes | PSNR after one multiplicative GT-mean alignment |
| `ssim` | SSIM | yes | `structural_similarity`, channel-averaged |
| `ssim_aligned` | SSIM (mean-aligned) | yes | SSIM after mean alignment |
| `lpips` | LPIPS | yes | `lpips.LPIPS(net='alex')`, inputs in [-1, 1] |
| `delta_e00` | Colour distortion | yes | Mean CIEDE2000 difference |
| `hue_shift_deg` | Hue shift | input/GT | Mean wrapped hue difference on `C* > 10` |
| `niqe` | NIQE | no | `pyiqa.create_metric('niqe')` |
| `noise_relative` | Relative noise | no | Immerkaer σ on flat 50% ÷ mean luminance (absolute noise, not a suppression ratio) |
| `noise_residual` | Noise residual | yes | Mean absolute residual in flat regions |
| `edge_preservation` | Edge preservation | input/GT | Gradient correlation on the strongest 10% edges |
| `texture_retention` | Texture retention | input/GT | Local-std ratio in the top 10% textured patches |
| `new_clipping` | New clipped pixels | input | `frac(out ≥ 0.98) − frac(in ≥ 0.98)` |
| `highlight_push` | Highlight push | input | Share of input in [0.85, 0.98] pushed to clipping |
| `halo_rate` | Halo rate | input | Gradient-reversal rate on the strongest 5% input edges |
| `loe` | LOE | input | Lightness Order Error on 50×50 max-RGB lightness maps |
| `black_crushing` | Black-level crushing | input | `frac(out max ≤ 0.02) − frac(in max ≤ 0.02)` |

Two thresholds are not fixed by the decision document and are therefore exposed
as parameters with documented defaults:

- `texture_retention` uses the 90th percentile of local standard deviation to
  define "textured patches".
- The GT-mean-aligned variants use `out × mean(gt) / mean(out)`, clipped to
  [0, 1].

## Harness

```python
from backend.evaluation import evaluate_pair, evaluate_set, ImagePair

values = evaluate_pair(
    "out.png",
    reference="gt.png",
    original="low.png",
)
values["psnr"].value, values["psnr"].notes

rows = evaluate_set([
    ImagePair(prediction="out/a.png", reference="gt/a.png", original="low/a.png"),
])
```

`evaluate_arrays` accepts arrays directly. Adding a metric means calling
`register_metric(MetricDefinition(...))`; the harness and CLI never need to
change.

## CLI

```bash
# JSON list of pairs
python -m backend.evaluation.cli --pairs pairs.json --output reports/eval

# Directory mode, matched by relative path
python -m backend.evaluation.cli \
    --predictions outputs/ --reference gt/ --original low/ \
    --output reports/eval --name "Hybrid vs GT"
```

`pairs.json`:

```json
[
  {"name": "img1", "prediction": "out/img1.png",
   "reference": "gt/img1.png", "original": "low/img1.png"}
]
```

Writes `evaluation.json`, `evaluation.csv` and `evaluation.md` containing
mean ± std, median, P10/P90 and the worst five images per metric. Pass
`--keys psnr,ssim` to restrict the run.

## Aggregation

`backend/evaluation/aggregate.py` provides the §6.6 requirements:
mean/std/median/P10/P90, worst-N, and a paired Wilcoxon signed-rank test
(SciPy). `select_hard_slice` accepts an externally supplied feature table
(`p50`, `nu`); LumenScan computes those features in a later phase, so this
module does not re-implement it.

## Latency and memory

`backend/evaluation/latency.py` implements §6.7: 10 warm-up runs, then timed
runs with fixed OpenCV/PyTorch thread counts, median/P95, per-stage averages and
peak RSS sampled with `psutil`.

```python
from backend.evaluation import LatencyBenchmark

result = LatencyBenchmark(process_one_image, warmups=10, runs=100, threads=4).run()
result.median_ms, result.p95_ms, result.per_stage_ms, result.peak_rss_mb
```

## Dependencies and lazy loading

- `requirements-eval.txt`: numpy, OpenCV, scikit-image, imagehash, SciPy,
  psutil.
- `requirements-metrics.txt`: torch, torchvision, lpips, pyiqa.
- Importing `backend.evaluation` (and therefore the API) does **not** import
  numpy or OpenCV. LPIPS and NIQE models are built on first use only.

LPIPS downloads a 233 MB AlexNet checkpoint and pyiqa may download NIQE model
files. Those downloads must never happen in the unit suite.

## Tests

Unit tests (default, network-free) mock model construction and verify the lazy
plumbing:

```bash
python -m pytest -q          # use the latest GitHub Actions run for the exact current count
```

Real LPIPS/NIQE tests are marked `integration`, excluded by `pytest.ini`, and
carry a 180-second timeout:

```bash
python -m pytest -m integration -o addopts="" tests/test_metrics_integration.py
```

Run the integration tests only when the model weights are already cached (or
when the download is acceptable).

## Kaggle evaluation

The Kaggle runner lives in `training/kaggle_eval/`. It discovers the permanent
dataset under `/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets`,
the ExDark and DICM mounts, and datasets transferred into `/kaggle/working`;
it evaluates original/classical/deep-learning/hybrid modes with the existing
pipeline and writes deterministic manifests, per-image records, aggregates,
latency, failures and before/after visuals. LIME, MEF, NPE/NPE-ex1/2/3 and VV
are unpaired; NPE and its extensions are one logical report with subgroup
metadata. Training and test records are both discovered for count verification;
only test samples are evaluated. See
[`training/kaggle_eval/README.md`](../training/kaggle_eval/README.md) and
`example_config.json`.


## Joint fork comparison protocol

For an A/B comparison, use the same dataset manifest, checkpoint file, model scale factor, execution device, precision, image preprocessing, warm-up count, timed runs and metric dependencies for every configuration. Record the Git commit SHA and the full environment settings with each run.

Recommended sequence:

1. Baseline: `LUMENOIR_RETINEX_DOWNSAMPLE=2`, `LUMENOIR_HUELOCK_SOURCE_ANCHOR=0`.
2. Retinex factor trial: test downsample factors 1, 2, 3 and 4 with source anchoring disabled.
3. HueLock anchor trial: keep the selected Retinex factor fixed; compare anchor strength 0 and 0.45.
4. Combined candidate: run the selected Retinex factor with anchor 0.45.

Use the repository's existing `training.kaggle_eval` runner to create per-image records and aggregate reports. The currently verified model file is the published pretrained Zero-DCE++ checkpoint. Compare it against the project's own fine-tuned `best.pth` when Colab training is complete. A separate collaborator checkpoint is optional; do not describe the published pretrained weights as a friend's fine-tuned model. Keep both files unchanged and select them explicitly through the checkpoint matrix runner. Do not treat previous LUMENOIR or collaborator-reported aggregate means as directly comparable until the same images and metric implementation have been rerun in this fork.
