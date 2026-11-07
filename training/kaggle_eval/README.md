# Kaggle evaluation

This package evaluates the already-implemented LUMENOIR pipeline on datasets
mounted under `/kaggle/input` and datasets transferred into `/kaggle/working`.
The permanent Kaggle dataset is
`/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets`; ExDark and
DICM use their own mounts; SICE/NPE/LIME/MEF/VV are transferred into
`/kaggle/working` (or an extracted temporary root) when needed. It does **not**
reimplement enhancement, tune parameters, fine-tune the model or download
datasets.

## Quick start

1. Attach the permanent dataset at
   `/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets`.
   ExDark is at `/kaggle/input/datasets/washingtongold/exdark-dataset`; DICM is
   at `/kaggle/input/datasets/twozeb/dicm-img`.
2. Transfer SICE and NPE/NPE-ex1/NPE-ex2/NPE-ex3, LIME, MEF and VV into
   `/kaggle/working` when they are needed. Do not assume an
   additional-test-datasets mount exists.
3. Attach the LUMENOIR repository (or clone it into `/kaggle/working/LUMENOIR`)
   and the Zero-DCE++ checkpoint (`zero_dce_plus_plus.pth` or `Epoch99.pth`).
4. Install the evaluation dependencies if they are not already present:

   ```bash
   pip install -r requirements-eval.txt
   pip install -r requirements-metrics.txt   # optional: LPIPS, NIQE, torch
   ```

5. Inspect the discovered layout before running:

   ```bash
   python -m training.kaggle_eval --discover-only \
     --input-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
     --additional-root /kaggle/input/datasets/washingtongold/exdark-dataset \
     --additional-root /kaggle/input/datasets/twozeb/dicm-img \
     --additional-root /kaggle/working \
     --additional-root /kaggle/working/lumenoir_additional/extracted \
     --output-dir /kaggle/working/lumenoir-evaluation
   ```

   Locally the equivalent roots are `datasets/raw`, the automatic
   `datasets/raw/additional_test_datasets/data`, and optionally
   `datasets/raw/additional_test_datasets`.

6. Copy `example_config.json`, edit the roots/pairs/weights, and run:

   ```bash
   python -m training.kaggle_eval \
     --config /kaggle/working/eval_config.json
   ```

The example notebook (`kaggle_notebook.ipynb`) performs the same steps.

## Dataset discovery and explicit configuration

Auto-discovery searches `/kaggle/input` first, then every configured additional
root. It proposes roots and low/high pair directories; it is a convenience, not
a guarantee:

- Roots are found under each search root and one nested level.
- LOL-v1 prefers `eval15`/test; LOL-v2 prefers `Test`; both train and test
  layouts are discovered for count verification. Only test/eval records are
  evaluated; training records are never passed to the runner.
- Pairs are matched by filename stem, with `low*`/`normal*` prefixes normalized
  (`low00690.png` ↔ `normal00690.png`).
- Missing references are reported and excluded; paired metrics are never
  computed without a reference.
- SICE supports `train/samples/*/{Input,GT}.jpg` and
  `test/samples/*/{Input,GT}.jpg`. Only test samples are evaluated; training
  samples are kept out of the evaluation set.
- Unpaired datasets are read recursively, filtered by image extension.
- JPG, JPEG, PNG and BMP are supported. Other known image formats (for example
  WebP, GIF or HEIC) are not silently dropped: discovery records a warning.

### Additional unpaired datasets

The extracted datasets are handled as four logical unpaired datasets; the
seven source folders are retained as subgroup information where applicable:

| Logical dataset | Local source path | Images | Format |
| --- | --- | ---: | --- |
| `lime` | `datasets/raw/additional_test_datasets/data/LIME/` | 10 | BMP |
| `mef` | `datasets/raw/additional_test_datasets/data/MEF/` | 17 | PNG |
| `npe` | `datasets/raw/additional_test_datasets/data/NPE/` | 8 | mixed |
| `npe` (subgroup `npe_ex1`) | `datasets/raw/additional_test_datasets/data/NPE-ex1/` | 29 | mixed |
| `npe` (subgroup `npe_ex2`) | `datasets/raw/additional_test_datasets/data/NPE-ex2/` | 18 | mixed |
| `npe` (subgroup `npe_ex3`) | `datasets/raw/additional_test_datasets/data/NPE-ex3/` | 30 | mixed |
| `vv` | `datasets/raw/additional_test_datasets/data/VV/` | 24 | mixed |

NPE total: **85** images. NPE and its three extensions produce one `npe`
manifest and one set of dataset-level aggregates; every record retains its
`subgroup` (`npe`, `npe_ex1`, `npe_ex2`, `npe_ex3`), and the manifest contains
per-subgroup counts. These four logical datasets are unpaired, so PSNR, SSIM
and LPIPS are never calculated for them.

If auto-discovery is ambiguous, set the dataset explicitly. The JSON structure
is:

```json
{
  "additional_roots": [
    "/kaggle/input/datasets/washingtongold/exdark-dataset",
    "/kaggle/input/datasets/twozeb/dicm-img",
    "/kaggle/working"
  ],
  "datasets": {
    "lol_v1": {
      "root": "/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets",
      "pairs": [{"low": "eval15/low", "high": "eval15/high", "split": "test"}]
    },
    "npe": {"root": "/kaggle/working"},
    "exdark": {"root": "/kaggle/input/datasets/washingtongold/exdark-dataset", "images": [""]}
  }
}
```

NPE auto-groups from the transferred `/kaggle/working` folders. LIME, MEF and
VV are discovered automatically from `/kaggle/working` or an extracted
additional root. No permanent additional-test dataset is assumed.

Training records are present in manifests for count verification but are never
returned by the evaluation runner. `allow_training_split` is retained as a
legacy configuration field but no longer enables evaluation on a training
split.

## Evaluation configurations

Two prepared configurations cover the controlled experiment:

- `configs/classical_baseline.json`: `original`, `classical`, `hybrid`
- `configs/deep_learning.json`: `original`, `deep_learning`, `hybrid`

Both include PSNR, SSIM, LPIPS, NIQE, CIEDE2000, noise, edge and latency. Paired
and unpaired results are written separately by dataset kind. Both expect the
Zero-DCE++ checkpoint at `/kaggle/working/zero_dce_plus_plus.pth` unless edited.

SICE is native-resolution and expensive. See
[`../../docs/sice_evaluation.md`](../../docs/sice_evaluation.md) and run it as a
separate batch with `--latency-runs 0`.

## Controlled pretrained-versus-fine-tuned checkpoint matrix

Use `training/kaggle_eval/compare_checkpoints.py` to compare one or more
strictly compatible Zero-DCE++ checkpoints under the same datasets and
pipeline settings. A valid single-checkpoint run is supported; the usual
research comparison uses:

- `pretrained`: the published Zero-DCE++ Epoch99 checkpoint
  (`zero_dce_plus_plus.pth`).
- `abhi`: the `best.pth` exported by the project's own fine-tuning run.

A separate collaborator checkpoint is optional. Do not describe the published
pretrained checkpoint as a collaborator's fine-tuned model unless that is what
the file actually is. Checkpoints are supplied as Kaggle input datasets and
are never committed to Git.

Example:

```bash
python -m training.kaggle_eval.compare_checkpoints \
  --config /kaggle/working/eval_config.json \
  --checkpoint pretrained=/kaggle/input/zero-dce-plus-plus/zero_dce_plus_plus.pth \
  --checkpoint legacy_epoch100=/kaggle/input/legacy-lumenoir/zero_dce_plus_plus_legacy_epoch100.pth \\
  --checkpoint abhi=/kaggle/input/abhi-lumenoir-finetuned/best.pth \
  --datasets lol_v1,lol_v2_real,lol_v2_syn \
  --device cuda \
  --output-dir /kaggle/working/lumenoir-joint-matrix
```

The runner calculates SHA-256 for every supplied checkpoint, runs the shared
Original/Classical reference once, then runs Deep Learning and Hybrid for each
checkpoint across the Retinex/HueLock matrix. The output
`matrix_manifest.json` records checkpoint labels, paths, hashes, settings,
counts, failures and report locations. The reference and model runs are
separate; a non-zero exit code means the matrix contains an error or no
successful model run and must be inspected before drawing conclusions.

## Leakage check

The read-only leakage audit consumes a manifest and never writes splits:

```bash
python -m training.kaggle_eval.leakage_check \
  --manifest /kaggle/working/lumenoir-evaluation/manifests/manifest.json \
  --output /kaggle/working/lumenoir-leakage.json
```

See [`../../docs/leakage.md`](../../docs/leakage.md) for the actual local result
and the warning that pHash/SSIM exclusions have not been applied to the
evaluation set without approval.

## Evaluation modes

| Mode | Description |
| --- | --- |
| `original` | The low-light input, evaluated directly (baseline). |
| `classical` | LumenScan + LumenRetinex + EdgeGuard + HueLock + alpha blend. |
| `deep_learning` | LumenScan + Zero-DCE++ + EdgeGuard + HueLock + alpha blend. |
| `hybrid` | Full pipeline. |

All modes use the existing `EnhancementPipeline`; no enhancement code is
duplicated. `deep_learning` and `hybrid` require a ready checkpoint and PyTorch.
If the model is unavailable and `allow_missing_model` is true, those modes are
skipped and logged in `failures/` and `run_config.json`.

### Shared computation across modes

For one image, the analysis (LumenScan), classical (LumenRetinex) and
deep-learning (Zero-DCE++) stages depend only on the image, not on the selected
mode. The runner therefore computes each of them once per image and replays the
measured result for the modes that require it. Fusion (LumenFuse) runs only for
hybrid and consumes the reused classical and deep-learning candidates. Only the
downstream stages (EdgeGuard, HueLock, alpha blend) run per mode, because their
input differs by mode.

The cache is per image and is released before the next image, so only one
image's intermediates are retained. Outputs and metric values are unchanged;
per-stage timings are replayed so each mode still reports the same stage-cost
attribution. A failure in a shared stage only affects modes that require that
stage. `share_stages` (default `true`) can be set to `false`, or
`--no-share-stages` passed on the CLI, to reproduce the previous behaviour for
an A/B comparison.

### Benchmarking the optimisation

`training.kaggle_eval.benchmark` runs the harness twice on the same images, once
with per-mode recomputation and once with shared stages, and writes
`benchmark.json` / `benchmark.md` with total runtime, per-mode recorded
processing time, execution counts, peak GPU memory and output equivalence
(metrics plus saved visuals):

```bash
python -m training.kaggle_eval.benchmark \
  --config /kaggle/working/eval_config.json --images 10
```

Run it on the same GPU, checkpoint and dataset configuration used for the real
evaluation. Report only measured numbers from that run.

## Metrics

Paired datasets (LOL-v1, LOL-v2 Real/Synthetic, SICE P1 with valid pairs):

- PSNR, SSIM, LPIPS, CIEDE2000 colour distortion.

Unpaired datasets (ExDark, DICM, NPE including NPE-ex1/2/3, LIME, MEF, VV):

- NIQE, LOE, relative noise, edge preservation, texture retention, hue shift,
  new clipping, highlight push, black-level crushing, halo rate.

Definitions and assumptions:

- **PSNR/SSIM** use the evaluation framework (`skimage`, `data_range=1.0`).
- **LPIPS** uses `lpips` with AlexNet; its pretrained weights may download on
  first use. Use `skip_metrics: ["lpips"]` if the environment is offline.
- **NIQE** uses `pyiqa`; compare relative values from this same code, not
  MATLAB numbers.
- **LOE** is the NPE-paper lightness-order error on 50×50 max-RGB maps: the
  fraction of valid pixel pairs whose relative lightness order changes. It
  requires the low-light input and is therefore reported for unpaired sets.
- **Noise** is `σ̂_flat / mean(Y_out[flat])` (absolute relative noise) from the
  Immerkaer estimator. It is not a suppression ratio; compare the original and
  enhanced modes (or the paired `noise_residual`) to assess suppression.
- **Edge preservation** is the Pearson correlation of gradient magnitudes on the
  strongest 10% of edges; unpaired comparisons use the input.
- **Colour distortion** is CIEDE2000 for paired sets and mean wrapped hue
  difference on `C* > 10` pixels for unpaired sets.
- **New clipping** is `frac(out max-channel ≥ 0.98) − frac(in max-channel ≥ 0.98)`.
- **Black-level crushing** is `frac(out max-channel ≤ 0.02) − frac(in max-channel ≤ 0.02)`.
- **Halo rate** is the gradient-reversal rate on the strongest 5% of input
  edges.
- **Latency** includes per-image total processing time, per-stage aggregates,
  and a warm-up benchmark (`latency_warmups`/`latency_runs`) for the first
  evaluation image of each dataset/mode.

Missing optional dependencies or a failed metric produce a `null` value with a
note in the per-image JSONL record, never a fabricated number.

## Outputs

Everything is written under the configured output directory, default
`/kaggle/working/lumenoir-evaluation`:

```
manifests/
  manifest.json                 # all datasets, sorted and deterministic
  <dataset>.json
  discovery_report.json         # image counts, format counts, subgroup counts
records/
  records.jsonl                 # image-level metric records + subgroup + notes + timings
  records.csv
reports/<dataset>/<mode>/
  image_metrics.json|.csv|.md   # per-image rows, summary, worst five
aggregate.json|.csv|.md         # dataset/mode/metric aggregates
latency.json|.csv|.md           # stage, total and warm-up benchmark latency
latency_totals.csv
failures/
  failures.jsonl|.csv
visuals/<dataset>/<image>_<mode>.png
device.json
run_config.json                 # config, seed, git commit, device, model status
```

Every manifest entry contains absolute image paths, `record_count`,
`image_count`, `evaluation_count`, `training_count`, `split_counts`,
`format_counts`, `subgroup_counts` and the source `roots`.
Visuals are side-by-side low-light/input and output images, capped by
`visual_limit` per dataset.

## Reproducibility

- Manifest records are sorted by relative path; reruns are deterministic.
- Both train and test records are present in manifests for count verification;
  the runner validates each manifest and evaluates only test/eval/unpaired
  records. Training records are never passed to the pipeline.
- `max_images` sampling uses `random.Random(seed)` and is recorded in the
  manifest warnings.
- `seed`, config, additional roots and git commit are in `run_config.json`.
- Per-image records, total processing time and per-stage timings are retained.
- No parameter is tuned on evaluation data.
- Counts are never inferred from filenames: manifests report the discovered
  image count, evaluation count, training count, split counts, per-format
  counts and NPE subgroup counts. Unsupported image formats, empty directories
  and missing NPE subgroups are warned about rather than silently ignored.

## Assumptions and limitations

- No dataset is downloaded. Paths must already exist under `/kaggle/input`,
  `/kaggle/input/datasets`, `/kaggle/working`, or be supplied explicitly.
- Unpaired datasets (including LIME, MEF, NPE/NPE-ex1/2/3 and VV) have no ground
  truth, so PSNR/SSIM/LPIPS are not reported.
- SICE Part 1 is not held out for Zero-DCE++ (the model was pre-trained on it);
  paired SICE results must be labelled accordingly.
- LOL and SICE training records are discovered for counts but excluded from
  evaluation. Explicit training paths do not enter the evaluation set.
- Paired images must have identical dimensions; mismatches are logged as
  failures rather than resized silently.
- SICE test images in the local Hugging Face copy are about 5997×3999; full
  SSIM/CIEDE2000/LOE evaluation and the warm-up benchmark can be slow. Run SICE
  in a separate batch, or set `--latency-runs 0` for that batch.
- LPIPS/NIQE require optional packages and model files; on a network-isolated
  Kaggle run, skip them or pre-cache the model files.
- Metrics are only as comparable as their library versions; keep the same
  environment across methods.
- No quality improvement claim should be made until the reports are populated
  with real datasets.
- GPU speeds up Zero-DCE++ only; LumenScan, LumenRetinex, LumenFuse, EdgeGuard
  and HueLock are CPU NumPy/OpenCV work.
