# LUMENOIR

**Robust low-light image restoration and enhancement.**

LUMENOIR is an illumination-adaptive image restoration workspace: upload a
severely underexposed, noisy or colour-distorted photograph, have it analysed,
and receive a restored result. It is designed to stay lightweight enough for
affordable CPU-based cloud deployment and responsive on phones, tablets and
desktops.

> **Status: full enhancement pipeline integrated, CPU and accelerator paths.**
> Upload validation, LumenScan, LumenRetinex, pretrained Zero-DCE++ inference,
> LumenFuse, EdgeGuard, HueLock, the adaptive alpha blend, the dataset pipeline,
> the evaluation framework, the **Kaggle evaluation runner** and the frontend
> workspace are implemented. **`POST /api/enhance` now returns an enhanced
> PNG.** The accelerator path runs every stage on CUDA/MPS tensors; CPU remains
> the reference and fallback. Classical mode works without a model;
> deep-learning and hybrid modes require a ready Zero-DCE++ checkpoint. Local datasets are present under
> `datasets/raw` (LOL-v1, LOL-v2 Real/Synthetic, SICE_Paired and the additional
> NPE/LIME/MEF/VV folders); ExDark and DICM are not local. No full evaluation
> has been run.

---

## Implemented vs. placeholder

| Area | Status |
| --- | --- |
| Image upload + validation (type, size, dimensions, corruption) | **Implemented** |
| Image information + basic statistics (mean/stddev/min/max, histogram) | **Implemented** |
| Safe aspect-preserving resizing for inference | **Implemented** |
| Pipeline orchestration, per-stage timing, failure isolation | **Implemented** |
| Model interface, registry, load-once manager, CPU config | **Implemented** (Zero-DCE++ registered) |
| Pretrained Zero-DCE++ (NoirNet) inference | **Implemented** (checkpoint load-verified) |
| Classical CV primitives (box/guided filter, Immerkaer noise, illumination map) | **Implemented** |
| LumenScan statistics + rule-based parameter selection | **Implemented** (wired into the pipeline) |
| LumenRetinex (adaptive gain, highlight protection, CLAHE on L\*) | **Implemented** (wired into the pipeline) |
| EdgeGuard (gain-aware luma + chroma denoising) | **Implemented** (wired into the pipeline) |
| HueLock (guarded LAB colour-cast correction) | **Implemented** (wired into the pipeline) |
| LumenFuse (noise-aware Laplacian-pyramid fusion) | **Implemented** (wired into the pipeline) |
| Full pipeline: scan → Retinex/DL → fuse → denoise → colour → α blend | **Implemented** |
| Adaptive alpha blend and do-no-harm bypass | **Implemented** |
| Device selection (CPU / CUDA / Apple Silicon MPS / auto) | **Implemented** |
| GPU-first pipeline (scan → Retinex/DL → fuse → denoise → colour → α blend) | **Implemented** (CUDA/MPS tensor path; CPU fallback) |
| GPU PSNR/SSIM evaluation | **Implemented** (`LUMENOIR_EVAL_DEVICE`) |
| Pipeline benchmark (CPU/accelerator, stage timings, memory) | **Implemented** (`training/benchmarks/benchmark_pipeline.py`; no T4 run yet) |
| Missing-weights, checkpoint, device and inference-failure handling | **Implemented** |
| API: health, limits, upload, pipeline status, enhance, metrics | **Implemented** |
| Frontend: upload, methods, comparison, intermediate gallery, downloads | **Implemented** |
| Evaluation framework (PSNR, SSIM, LPIPS, NIQE, artifact metrics, latency) | **Implemented** (no benchmark run yet) |
| Kaggle evaluation runner (discovery, four modes, reports, visuals) | **Implemented** (local discovery verified; no full evaluation run) |
| Dataset preparation pipeline (validation, pHash dedupe, leakage, splits, crops) | **Implemented** (local data present; real splits not generated) |
| Retinex / CLAHE / denoising / colour correction | **Implemented** |
| Deep learning architecture, weights, inference | **Implemented** (pretrained Zero-DCE++, wired into the pipeline) |
| Adaptive fusion, fusion weights, parameter selection | **Implemented** |
| Training notebooks and fine-tuning | **Placeholder** (folder + notes only) |

`POST /api/enhance` returns `200` with a base64 PNG for implemented modes.
Classical mode works without a model; `deep_learning` and `hybrid` return `503`
until `LUMENOIR_MODEL_ID=zero_dce_plus_plus` is set and the checkpoint and
inference dependencies are available.

---

## Folder structure

```
LUMENOIR/
├── context.md                     # project brief + implementation status
├── README.md                      # this file
├── requirements.txt               # backend runtime dependencies
├── requirements-dev.txt           # test / lint dependencies
├── ruff.toml                      # Python lint configuration
├── Dockerfile                     # backend-only image
├── .env.example                   # documented environment variables
│
├── backend/
│   ├── main.py                    # app factory, lifespan, error handlers
│   ├── config.py                  # upload limits, inference settings, CORS
│   ├── timing.py                  # per-stage stopwatch
│   ├── api/
│   │   ├── deps.py                # settings / model manager / pipeline
│   │   ├── schemas.py             # request/response models
│   │   ├── serializers.py         # shared image + statistics serialisation
│   │   ├── routes_health.py       # GET /api/health
│   │   ├── routes_images.py       # limits + validated upload
│   │   ├── routes_enhance.py      # status + enhancement request
│   │   └── routes_evaluation.py   # declared metrics
│   ├── preprocessing/
│   │   ├── validation.py          # upload validation + single decode
│   │   ├── loader.py              # standalone decode helper
│   │   ├── statistics.py          # descriptive measurements
│   │   └── resize.py              # safe inference sizing
│   ├── inference/                 # model-agnostic plumbing
│   │   ├── base.py                # EnhancementModel interface
│   │   ├── registry.py            # model registry (zero_dce_plus_plus)
│   │   ├── manager.py             # load-once manager, timing, errors
│   │   ├── device.py              # cpu / mps / cuda / auto resolution
│   │   └── errors.py              # inference error types
│   ├── enhancement/
│   │   ├── pipeline.py            # orchestrator + capability reporting
│   │   ├── stages.py              # seven implemented stages over the modules
│   │   └── types.py               # modes, stages, artifacts, results
│   ├── classical_cv/
│   │   ├── arrays.py              # validation, luma, smoothstep
│   │   ├── filters.py             # box + guided filters
│   │   ├── noise.py               # Immerkaer sigma + local noise map
│   │   ├── illumination.py        # LIME-style illumination map
│   │   ├── colour.py              # LAB colour-cast detection
│   │   ├── scan.py                # LumenScan statistics + parameters
│   │   ├── retinex.py             # LumenRetinex gain, shoulder, CLAHE
│   │   ├── edge_guard.py          # gain-aware luma + chroma denoising
│   │   └── hue_lock.py            # guarded LAB colour-cast correction
│   ├── deep_learning/
│   │   ├── zero_dce_net.py        # Zero-DCE++ architecture (torch)
│   │   └── zero_dce_plus_plus.py  # wrapper, checkpoint validation, registration
│   ├── fusion/
│   │   └── lumen_fuse.py          # noise-aware Laplacian-pyramid fusion
│   ├── evaluation/                # metric framework
│   │   ├── metrics.py             # declarations + lightweight adapter
│   │   ├── reference_metrics.py   # PSNR, SSIM, LPIPS, CIEDE2000
│   │   ├── no_reference_metrics.py# NIQE, noise, edges, colour, clipping, halo
│   │   ├── harness.py             # evaluate_pair / evaluate_set
│   │   ├── aggregate.py           # mean/std/median/P10/P90, Wilcoxon
│   │   ├── latency.py             # warmups, P50/P95, peak RSS
│   │   └── cli.py                 # JSON/CSV/Markdown reports
│   └── weights/                   # .pth checkpoints (git-ignored; Zero-DCE++ present)
│
├── frontend/                      # React + TypeScript + Tailwind CSS (Vite)
│   ├── src/
│   │   ├── App.tsx                # shell, header, view switching
│   │   ├── pages/                 # LandingPage, WorkspacePage
│   │   ├── components/            # dropzone, slider, method selector,
│   │   │                          # intermediate gallery, info card
│   │   ├── lib/                   # api client, formatting, downloads
│   │   └── assets/                # two small SVG placeholders
│   └── vite.config.ts             # Tailwind plugin + /api dev proxy
│
├── tests/                         # backend, dataset pipeline and metric tests
├── docs/
│   ├── architecture.md            # implemented vs placeholder, lifecycle
│   ├── api.md                     # endpoints and every error code
│   ├── datasets.md                # expected tree, sources, split protocol
│   ├── evaluation.md              # metrics, harness, integration tests
│   ├── classical_cv.md            # primitives, LumenScan, Retinex, EdgeGuard, HueLock
│   ├── fusion.md                  # LumenFuse weights, pyramids, limitations
│   └── zero_dce.md                # NoirNet provenance, devices, timings
│
├── training/
│   ├── data_pipeline/             # dataset preparation package
│   ├── kaggle_eval/               # Kaggle discovery, runner, notebook, samples
│   ├── evaluate.py                # evaluation CLI wrapper
│   ├── splits/                    # committed split spec / split files
│   └── reports/                   # committed cleaning reports
└── assets/                        # PLACEHOLDER — future diagrams
```

`frontend/dist`, `.venv`, `node_modules`, model weights, datasets and `.env`
files are git-ignored.

---

## Setup and run

Prerequisites: Python 3.11+ (verified on 3.14), Node 20+ (verified on 22), and
Git. Two terminals are used for development.

### 1. Backend (FastAPI)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

uvicorn backend.main:app --reload --port 8000
```

- API: <http://localhost:8000/api/health>
- Interactive docs: <http://localhost:8000/docs>

### 2. Frontend (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

- App: <http://localhost:5173>

The dev server proxies `/api/*` to `http://127.0.0.1:8000`, so no API URL needs
to be configured. To point the UI at a different backend, set
`VITE_API_BASE_URL` before building.

### 3. Optional: pretrained Zero-DCE++ inference

The API runs without these; installing them enables NoirNet.

```bash
pip install -r requirements-inference.txt   # torch + numpy
LUMENOIR_MODEL_ID=zero_dce_plus_plus \
LUMENOIR_INFERENCE_DEVICE=cpu \
uvicorn backend.main:app --reload --port 8000
```

The checkpoint is already at `backend/weights/zero_dce_plus_plus.pth`
(git-ignored). With the model loaded, `GET /api/enhance/status` reports
`model.state = ready`, and `POST /api/enhance` returns images for
`deep_learning` and `hybrid` as well as `classical`. Without the model or
PyTorch, only classical mode is available and the other two return `503`.
Device values: `cpu`, `mps`, `cuda`, `cuda:N`, `auto`. See
[`docs/zero_dce.md`](docs/zero_dce.md) for provenance, verification and
measured timings.

### 4. Tests and checks

```bash
pip install -r requirements-dev.txt        # runtime + eval + test tooling
.venv/bin/python -m pytest -q              # 317 tests, 8 integration deselected
.venv/bin/python -m ruff check backend tests training

cd frontend && npm run build && npm run lint
```

Integration tests are excluded because they use pretrained weights. Run them
deliberately:

```bash
# real Zero-DCE++ checkpoint (CPU/MPS) and the full API pipeline
.venv/bin/python -m pytest -m integration -o addopts="" \
  tests/test_zero_dce_integration.py tests/test_pipeline_integration.py

# LPIPS / NIQE (downloads a 233 MB AlexNet checkpoint on first use)
.venv/bin/python -m pytest -m integration -o addopts="" tests/test_metrics_integration.py
```

### 5. Dataset pipeline and evaluation

```bash
python -m training.data_pipeline check --sources   # works with no data
python -m training.data_pipeline split --spec-only # write the count specification
python -m training.data_pipeline check             # after downloading datasets
python -m training.data_pipeline all               # validate, dedupe, split, report

python -m training.evaluate --pairs pairs.json --output reports/eval
```

Local datasets are present under `datasets/raw`; ExDark and DICM are not local.
See [`docs/datasets.md`](docs/datasets.md)
for the exact folder tree, the sources to download manually and the split
protocol, and [`docs/evaluation.md`](docs/evaluation.md) for the metric
definitions and report format.

### 6. Kaggle evaluation

The Kaggle runner discovers the permanent dataset under
`/kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets`, the ExDark
and DICM mounts, and datasets transferred into `/kaggle/working`, evaluates the
four modes (original, classical, Zero-DCE++ only, hybrid) with the existing
pipeline, and writes deterministic manifests, per-image metrics, aggregates,
latency, failures and before/after visuals. LIME, MEF, NPE/NPE-ex1/2/3 and VV
are unpaired; NPE and its extensions are one logical `npe` report with subgroup
metadata. LOL/SICE training records are discovered for counts but only test
samples are evaluated.

```bash
# Inspect the proposed dataset layout first
python -m training.kaggle_eval --discover-only \
  --input-root /kaggle/input/datasets/chiranth25x/lumenoir-low-light-datasets \
  --additional-root /kaggle/input/datasets/washingtongold/exdark-dataset \
  --additional-root /kaggle/input/datasets/twozeb/dicm-img \
  --additional-root /kaggle/working \
  --additional-root /kaggle/working/lumenoir_additional/extracted \
  --output-dir /kaggle/working/lumenoir-evaluation

# Edit and run an explicit configuration
python -m training.kaggle_eval --config /kaggle/working/eval_config.json
```

See [`training/kaggle_eval/README.md`](training/kaggle_eval/README.md) for the
config format, outputs, metric definitions, limitations, and
[`training/kaggle_eval/kaggle_notebook.ipynb`](training/kaggle_eval/kaggle_notebook.ipynb)
for a ready-to-run notebook. `example_config.json` and sample report formats are
in the same directory.

### 7. GPU-first pipeline and benchmark

```bash
# CUDA deployment (T4 or similar)
LUMENOIR_DEVICE=cuda LUMENOIR_PRECISION=float32 \
LUMENOIR_INFERENCE_DEVICE=cuda LUMENOIR_MODEL_ID=zero_dce_plus_plus \
python -m uvicorn backend.main:app

# Full CPU/accelerator benchmark (writes JSON; no fabricated device rows)
python -m training.benchmarks.benchmark_pipeline \
    --devices cpu,cuda --sizes 256x256,640x480,1280x960,2304x1728 \
    --warmup 1 --repeat 3
```

See [`docs/gpu_pipeline.md`](docs/gpu_pipeline.md) for the stage mapping,
parity tolerances, measured local numbers and the pending T4 validation.

### 8. Docker (backend only)

```bash
docker build -t lumenoir-api .
docker run --rm -p 8000:8000 lumenoir-api
```

The frontend is a static build (`frontend/dist`) and is deployed separately.

---

## Configuration

Environment variables are read at startup and can be placed in `.env`
(see `.env.example`; `.env` is git-ignored):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LUMENOIR_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Maximum accepted upload size |
| `LUMENOIR_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |
| `LUMENOIR_MODEL_ID` | *(empty)* | Registered model id; `zero_dce_plus_plus` enables NoirNet; empty means none |
| `LUMENOIR_MODEL_WEIGHTS` | *(empty)* | Explicit checkpoint path |
| `LUMENOIR_WEIGHTS_DIR` | `backend/weights` | Where `.pth` files are looked up |
| `LUMENOIR_DEVICE` | `auto` | Whole-pipeline backend: `auto` (CUDA else CPU), `cpu`, `mps`, `cuda`, `cuda:N` |
| `LUMENOIR_PRECISION` | `float32` | `float32` or `float16` on CUDA/MPS; CPU is always float32 |
| `LUMENOIR_INFERENCE_DEVICE` | follows `LUMENOIR_DEVICE` | `cpu`, `mps`, `cuda`, `cuda:N` or `auto` (MPS → CUDA → CPU) |
| `LUMENOIR_INFERENCE_PRECISION` | follows `LUMENOIR_PRECISION` | Zero-DCE++ tensor precision |
| `LUMENOIR_EVAL_DEVICE` | `cpu` | GPU PSNR/SSIM: `cuda`, `cuda:N`, `mps` or `auto` |
| `LUMENOIR_ZERODCE_SCALE_FACTOR` | `12` | Zero-DCE++ downscale factor, 1–12 |
| `LUMENOIR_INFERENCE_MAX_EDGE` | `1600` | Longest edge used for inference |
| `LUMENOIR_INFERENCE_MAX_PIXELS` | `4000000` | Hard ceiling before a request is rejected |
| `LUMENOIR_LOAD_MODEL_ON_STARTUP` | `true` | Load weights once during startup |
| `VITE_API_BASE_URL` | *(empty = same origin)* | Frontend API base URL |
| `VITE_API_PROXY_TARGET` | `http://127.0.0.1:8000` | Dev proxy target |

**No API keys or credentials are required, and none are committed.** `.env` and
`.env.*` are git-ignored.

---

## API (current)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Service status, model state and readiness |
| `GET` | `/api/images/limits` | Accepted formats, size and dimension limits |
| `POST` | `/api/images/upload` | Validates an image, returns information + statistics |
| `GET` | `/api/enhance/status` | Stage, mode, model and artifact availability |
| `POST` | `/api/enhance` | Validates and runs the full pipeline; returns a base64 PNG (or `503` without a ready model for DL/hybrid) |
| `GET` | `/api/evaluation/metrics` | Implemented metrics, reference requirements and dependencies |

Uploads are validated and discarded; **no image is stored**. Rejections return
`{ "ok": false, "error": { "code": "...", "message": "..." } }`.

Full details, including every error code: [`docs/api.md`](docs/api.md).

---

## Zero-DCE++ (NoirNet) inference

The selected deep-learning backbone is implemented as a registered model:

- Registration: `@register_model("zero_dce_plus_plus")` in
  `backend/deep_learning/zero_dce_plus_plus.py`; `backend/main.py` imports the
  package so the registry is populated without changing any route or schema.
- Checkpoint: `backend/weights/zero_dce_plus_plus.pth` (52,395 bytes,
  git-ignored). Loading is strict: 10,561 parameters, 28 tensors, exact key set.
  A mismatched or original Zero-DCE checkpoint is rejected with
  `checkpoint_invalid`.
- Devices: `cpu`, `mps`, `cuda`, `cuda:N`, `auto`; explicit devices never fall
  back silently.
- Geometry: input is padded to a multiple of the scale factor and the output is
  cropped back to the exact input size.
- Torch and NumPy are imported only when the model loads, so the API still
  starts without them.

Provenance, verification hashes, the measured CPU/MPS timings and the
fine-tuning prerequisites are in [`docs/zero_dce.md`](docs/zero_dce.md).

Future architectures still use the same seam: add a module under
`backend/deep_learning/`, subclass `EnhancementModel`, register it, and point
`LUMENOIR_MODEL_ID` at it. The pipeline now consumes the registered model
through `ModelManager`; no route or schema change is needed.

### Measured pipeline latency (this machine)

Apple Silicon, Python 3.14, torch 2.15.0, CPU, model loaded once, hybrid mode,
one synthetic image per size. 512×512 is the median of three runs after a
warm-up; 1280×720 and 1920×1080 are single runs. These are observations, not a
general benchmark.

| Size | Total | Analysis | Retinex | Zero-DCE++ | Fusion | EdgeGuard | HueLock | Final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512×512 | 206.9 ms | 16.1 | 24.4 | 9.2 | 123.3 | 19.7 | 6.9 | 2.2 |
| 1280×720 | 774.8 ms | 49.4 | 92.0 | 37.2 | 508.0 | 58.6 | 21.9 | 7.6 |
| 1920×1080 | 1773.6 ms | 123.4 | 208.7 | 90.0 | 1158.1 | 132.3 | 44.5 | 16.5 |

Model load (including the first torch import) took about 1.2 s at startup.

---

## Validation rules (implemented)

- Accepted types: JPG, JPEG, PNG (extension **and** real decoded format must
  agree, so a renamed file is rejected).
- Max size: 10 MB by default.
- Dimensions: 64×64 minimum, 8000×8000 maximum, 40 MP maximum for upload.
- `POST /api/enhance` additionally enforces `LUMENOIR_INFERENCE_MAX_PIXELS`
  (default 4 MP) and returns `422 image_too_large` above it. The upload
  endpoint still accepts larger files; the processing ceiling protects the
  pipeline's memory and latency.
- Corrupted, truncated or non-image payloads are rejected with a specific code.
- Inference sizing never upscales and preserves aspect ratio; the original
  resolution is restored after processing. The current pipeline processes at
  the input resolution.
- Validation happens in memory; uploaded bytes are never written to disk.

---

## Deliberately not implemented

These decisions belong to research and remain open — nothing here guesses at
them, not even a temporary version:

- model training, fine-tuning, checkpoint export
- quality-based output selection across multiple complete runs, the closed-loop
  proxies and Optuna coefficient tuning
- region-aware enhancement, additional overexposure protection beyond the
  implemented gain/shoulder/original-candidate layers
- ablation runs, benchmark tables and deployment-target choice
- actual train/validation/test files (the rules are implemented; they are
  generated only once the datasets are downloaded)

The full enhancement pipeline, all six modules, the alpha blend and the API
integration are implemented. The closed-loop proxies (§5.7), Optuna tuning
(§5.6), dataset evaluation and fine-tuning are not.

Also intentionally absent: database, authentication, image storage and any
background queue. The runtime image now installs NumPy and OpenCV because the
pipeline needs them; Zero-DCE++ additionally needs
`requirements-inference.txt` (torch). Torch is still optional: the API starts
without it and disables deep-learning/hybrid modes. The API does not import
torch at startup.

---

## Next steps

1. Download the datasets listed in `docs/datasets.md` and run
   `python -m training.data_pipeline all` to produce the real split files.
2. Run the Kaggle evaluation runner (or local equivalent) on LOL-v1, LOL-v2,
   SICE, ExDark and the no-reference sets; inspect the generated reports.
3. Run ablations A–J, then fine-tune NoirNet and re-run every table.
4. Benchmark latency/memory and choose the deployment platform.


## Joint integration experiments

This repository is the user-owned fork of [Chiranthcommits25/Lumenoir](https://github.com/Chiranthcommits25/Lumenoir). The upstream repository is not modified by this project. Its API, React frontend, CPU implementation, CUDA/MPS tensor path, model manager, dataset pipeline and Kaggle evaluation runner are retained.

Two optional research controls are added without changing baseline defaults:

- `LUMENOIR_RETINEX_DOWNSAMPLE=2`: selects the illumination-map downsample divisor for both CPU and tensor/GPU Retinex. Valid values are 1–4. The existing divisor of 2 remains the default; the 0.33× result from the separate LUMENOIR implementation is not assumed to transfer to this LIME-style Retinex.
- `LUMENOIR_HUELOCK_SOURCE_ANCHOR=0`: enables an optional confidence-weighted source-chroma anchor in HueLock. It is off by default. Test `0.45` as an experimental setting; colourful, dark and noisy source regions receive reduced confidence.

The CPU and tensor paths use the same configuration values. Diagnostics include anchor weight and coverage. The controls must be evaluated on identical checkpoints, image splits, device, precision and metric definitions before any default is changed. See [the joint experiment protocol](docs/joint-integration.md).
