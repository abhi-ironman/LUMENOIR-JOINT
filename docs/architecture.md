# Architecture — current state

This document describes **what is actually implemented**, not the intended
research system. Anything not listed as implemented is a placeholder.

## Implemented

| Area | Location | Behaviour |
| --- | --- | --- |
| HTTP API | `backend/main.py`, `backend/api/` | FastAPI app, lifespan, CORS, shared error envelopes |
| Upload validation | `backend/preprocessing/validation.py` | Type, size, dimensions and integrity checks; one decode per request |
| Safe decoding | `backend/preprocessing/validation.py` | Detached RGB image, alpha removed, file handle closed |
| Basic statistics | `backend/preprocessing/statistics.py` | Per-channel mean/min/max/stddev + 16-bin luminance histogram |
| Safe resizing | `backend/preprocessing/resize.py` | Aspect-preserving, longest-edge cap, never upscales |
| Stage timing | `backend/timing.py` | `perf_counter` measurements per stage |
| Pipeline orchestration | `backend/enhancement/pipeline.py` | Ordered stages, per-mode stage sets, timing, failure isolation |
| Stage implementations | `backend/enhancement/stages.py` | Thin wrappers over LumenScan, LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard, HueLock and the alpha blend; CPU arrays or device tensors |
| Pipeline runtime | `backend/runtime/config.py` | Unified `LUMENOIR_DEVICE`/`LUMENOIR_PRECISION` resolution, CPU fallback for `auto`, explicit-device errors |
| Torch primitives | `backend/runtime/torch_ops.py` | LAB LUT, CLAHE, resize, filters, noise, guided filter, pyramids, colour spaces, bilateral |
| GPU classical stages | `backend/classical_cv/torch_*.py`, `backend/fusion/lumen_fuse.py` | Tensor implementations of scan, Retinex, EdgeGuard, HueLock and fusion; shared rules/parameters |
| GPU evaluation | `backend/evaluation/gpu_metrics.py` | Torch PSNR/SSIM selected by `LUMENOIR_EVAL_DEVICE`; CPU skimage default |
| Pipeline benchmark | `training/benchmarks/benchmark_pipeline.py` | CPU/CUDA/MPS wall time, stage medians, RSS and accelerator memory |
| Model interface | `backend/inference/base.py` | `preprocess → infer → postprocess` template + output validation |
| Model registry | `backend/inference/registry.py` | `register_model` / `create_model`; registers `zero_dce_plus_plus` |
| Model manager | `backend/inference/manager.py` | Load-once at startup, device resolution, missing-weights handling, serial inference, timing |
| Device resolver | `backend/inference/device.py` | `cpu` / `mps` / `cuda[:N]` / `auto`; explicit devices never fall back |
| Zero-DCE++ inference | `backend/deep_learning/zero_dce_plus_plus.py`, `zero_dce_net.py` | Pretrained NoirNet wrapper, strict checkpoint validation, padding/crop inference |
| Configuration | `backend/config.py` | Upload limits, inference paths/device/sizing, CORS |
| Evaluation registry | `backend/evaluation/metrics.py` | 12 metrics, all `implemented`; dependency-light declarations |
| Metric framework | `backend/evaluation/` | PSNR/SSIM/LPIPS/NIQE, noise, edge, colour, clipping, black crushing, LOE, halo, latency, aggregation, reports |
| Dataset pipeline | `training/data_pipeline/` | Discovery, validation, pHash dedupe, leakage, splits, crops, synthetic darks, reports |
| Kaggle evaluation | `training/kaggle_eval/` | Multi-root explicit config + discovery, four modes, NPE subgroup grouping, deterministic manifests/records/aggregates/latency/failures/visuals |
| Classical CV primitives | `backend/classical_cv/{filters,noise,illumination,colour}.py` | Box/guided filters, Immerkaer noise and map, illumination map, LAB cast detection |
| LumenScan | `backend/classical_cv/scan.py` | §5.2 features and §5.3 rule-based parameters; wired into the pipeline |
| LumenRetinex | `backend/classical_cv/retinex.py` | §3.3 adaptive gain, highlight shoulder, §3.4 CLAHE; wired into the pipeline |
| EdgeGuard | `backend/classical_cv/edge_guard.py` | §3.5 gain-aware luma guided filter + chroma bilateral; wired into the pipeline |
| HueLock | `backend/classical_cv/hue_lock.py` | §3.7 guarded LAB cast correction; wired into the pipeline |
| LumenFuse | `backend/fusion/lumen_fuse.py`, `backend/fusion/torch_backend.py` | §4 noise-aware weights, guided refinement and Laplacian-pyramid fusion; shared core with OpenCV CPU and PyTorch CUDA backends |
| Frontend shell | `frontend/src/App.tsx` | Header, view switching, footer |
| Landing page | `frontend/src/pages/LandingPage.tsx` | Product overview and honest status |
| Upload workspace | `frontend/src/pages/WorkspacePage.tsx` | Upload, validation, methods, results, downloads |
| Comparison component | `frontend/src/components/ComparisonSlider.tsx` | Pointer + keyboard slider, aspect-ratio aware |
| Method selector | `frontend/src/components/MethodSelector.tsx` | Classical / deep learning / hybrid, availability badges |
| Intermediate gallery | `frontend/src/components/IntermediateGallery.tsx` | Renders only outputs the backend actually produced |
| API client | `frontend/src/lib/api.ts` | Typed calls, structured errors, pipeline types |
| Tests | `tests/` | Validation, statistics, resize, inference, pipeline, API |

## Pipeline stages (implemented)

| Stage | Delegates to | Behaviour |
| --- | --- | --- |
| `analysis` | LumenScan | Features, illumination and noise maps, rule-based parameters, do-no-harm bypass decision |
| `classical_cv` | LumenRetinex | Adaptive gain, highlight shoulder, CLAHE on L\* → candidate C |
| `deep_learning` | Zero-DCE++ through ModelManager | Curve-network inference → candidate D; skipped when the bypass gate fires |
| `fusion` | LumenFuse | Pixel-wise noise-aware weights, guided refinement, Laplacian-pyramid blend → F; NumPy CPU or torch tensor backend |
| `denoising` | EdgeGuard | Gain-aware luma guided filter and chroma bilateral filter → F′ |
| `colour_correction` | HueLock | Guarded LAB cast correction → F″ |
| `quality_assessment` | Adaptive alpha blend | `out = (1−α)I + αF″`, or the original when α = 0 / bypass; final size/range validation |

The stage classes are orchestration only; algorithm decisions stay in the
modules. `PendingStage` remains available for custom partial pipelines but is
not used by `default_stages()`.

LumenFuse has one shared algorithm implementation in `fusion/lumen_fuse.py` and
two primitive backends: the original NumPy/OpenCV CPU path and a PyTorch path in
`fusion/torch_backend.py`. The unified pipeline resolves one runtime for the
whole request (`LUMENOIR_DEVICE`, default `auto`): when it resolves to CUDA or
MPS, every stage uses device tensors through `backend/runtime/torch_ops.py`;
otherwise the CPU reference path runs. `fuse()` still accepts the legacy
`LUMENOIR_FUSE_BACKEND`/`LUMENOIR_FUSE_DEVICE`/`LUMENOIR_FUSE_PRECISION`
variables for standalone callers and benchmarks. Torch is imported lazily, so
importing the pipeline or API never requires PyTorch and CPU tests stay
torch-free. Full details: [`gpu_pipeline.md`](gpu_pipeline.md).

## Not implemented

| Area | Status |
| --- | --- |
| Closed-loop proxy passes (§5.7) | Not implemented |
| Optuna coefficient tuning (§5.6) | Not implemented |
| Training and fine-tuning | Placeholder notebooks/config |
| Quality-based selection across multiple complete runs | Not implemented |
| Dataset evaluation and ablation tables | Not run (no dataset locally) |
| Kaggle evaluation run | Not run (runner implemented; attach datasets under `/kaggle/input`) |
| Model weights in Git | Checkpoint present locally, git-ignored |

## Model lifecycle (implemented plumbing)

```
create_app(settings)
  ├─ ModelManager(settings.inference)      # no model selected by default
  └─ EnhancementPipeline(model_manager=…)
        │
startup (lifespan)
  └─ manager.load_at_startup()
        ├─ no model_id            -> state = not_selected
        ├─ weights file missing   -> state = weights_missing   (app keeps running)
        ├─ device unavailable     -> state = failed            (app keeps running)
        ├─ checkpoint mismatch    -> state = failed            (app keeps running)
        ├─ factory raises         -> state = failed            (app keeps running)
        └─ model.load()           -> state = ready             (once per process)

request (POST /api/enhance)
  └─ run_in_threadpool(pipeline.run)
        ├─ runtime resolved lazily (LUMENOIR_DEVICE/PRECISION)
        ├─ CPU   → NumPy/OpenCV stages, manager.infer(PIL)
        └─ CUDA/MPS → TorchImageOps, all stages on device tensors,
                       manager.infer_tensor(tensor), stage sync for timing
```

The selected architecture, Zero-DCE++, is registered in
`backend/deep_learning/zero_dce_plus_plus.py`:

```python
@register_model("zero_dce_plus_plus")
class ZeroDCEPlusPlusModel(EnhancementModel):
    ...
```

`backend/main.py` imports `backend.deep_learning` so that registration happens,
and `LUMENOIR_MODEL_ID=zero_dce_plus_plus` plus the `.pth` file in
`backend/weights/` is all the configuration required. The model module imports
torch lazily, so the API still starts when PyTorch is absent; deep-learning and
hybrid modes are then reported unavailable and requests return `503`.

## Measured pipeline latency

Apple Silicon, Python 3.14, torch 2.15.0, CPU, model loaded once, hybrid mode,
synthetic images. 512×512 is the median of three runs after a warm-up;
1280×720 and 1920×1080 are single runs. These are observations for this machine,
not a general benchmark.

| Size | Total | Analysis | Retinex | Zero-DCE++ | Fusion | EdgeGuard | HueLock | Final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512×512 | 206.9 ms | 16.1 | 24.4 | 9.2 | 123.3 | 19.7 | 6.9 | 2.2 |
| 1280×720 | 774.8 ms | 49.4 | 92.0 | 37.2 | 508.0 | 58.6 | 21.9 | 7.6 |
| 1920×1080 | 1773.6 ms | 123.4 | 208.7 | 90.0 | 1158.1 | 132.3 | 44.5 | 16.5 |

Model load, including the first torch import, was about 1.2 s at startup.

## Request flow — enhance

```
Browser (React)
  │  POST /api/enhance  (file, mode)
  ▼
validate_and_decode()          type → content type → size → decode → format →
  │                            dimensions/megapixels
  ├─ failure → 4xx { ok:false, error:{ code, message } }
  ▼
compute_statistics()           descriptive measurements only
  ▼
run_in_threadpool(pipeline.run)
  ├─ analysis  → LumenScan + bypass decision
  ├─ classical → LumenRetinex
  ├─ DL        → Zero-DCE++ through ModelManager (serialised)
  ├─ fusion    → LumenFuse
  ├─ denoising → EdgeGuard
  ├─ colour    → HueLock
  └─ final     → α blend + size/range validation
  ├─ completed  → 200 { image, pipeline, output: base64 PNG }
  ├─ model/device unavailable → 503 typed error envelope
  └─ stage failure            → 500 enhancement_failed
```

The pipeline executes in a worker thread so the event loop is not blocked.
Classical-only requests can run concurrently; `ModelManager` serialises model
inference. The pipeline is stateless: every request gets a fresh context and no
image is stored.

## Intermediate outputs

The analysis stage publishes the original, illumination map and noise map. The
classical, deep-learning and fusion stages publish their outputs when they run;
fusion also publishes the enhancement-strength map. The final stage publishes
the delivered image. When the do-no-harm bypass fires, only the original, the
two analysis maps and the final output are available. Unavailable artifacts
carry a reason; the frontend renders only artifacts that were actually
produced.

## Module boundaries

- Routes stay thin; validation lives in `preprocessing`, orchestration in
  `enhancement`, model lifecycle in `inference`.
- Processing modules never import from `api`.
- `preprocessing/validation.py` is the only place that decodes uploads.
- `preprocessing/resize.py` is the only place that changes image size.
- `classical_cv/filters.py` is the only box/guided-filter implementation.
- `classical_cv/noise.py` is the only Immerkaer implementation; the evaluation
  noise module re-exports it.
- `classical_cv/scan.py` is the only place that selects LumenScan parameters;
  its rules and starting coefficients live in `LumenScanConfig`.
- `classical_cv/retinex.py` is the classical enhancement branch (gain,
  highlight shoulder, CLAHE); it consumes LumenScan parameters and does not
  choose them.
- `classical_cv/edge_guard.py` is the post-fusion denoiser; it consumes the
  LumenScan noise estimate and the original image.
- `classical_cv/hue_lock.py` applies the guarded cast shift using LumenScan's
  detected cast and raw beta.
- `fusion/lumen_fuse.py` computes the noise-aware weights, refines them with a
  guided filter and blends the three candidates in a Laplacian pyramid.
- `enhancement/stages.py` only sequences modules and publishes artifacts;
  `enhancement/pipeline.py` only schedules, times and reports.
- `evaluation/metrics.py` is the only place that declares metrics.
- Request handling is stateless: no database, no storage, no session.

## Deliberate omissions

- No database, no authentication, no external services.
- No image storage or history.
- NumPy and OpenCV are runtime dependencies because the pipeline needs them.
  PyTorch is optional and only loaded for Zero-DCE++; `lpips` and `pyiqa`
  remain evaluation-only.
- No untuned algorithm defaults beyond the starting coefficients documented in
  `final_decision.md` §4-§5; they live in `LumenScanConfig`, `LumenFuseConfig`
  and the other typed configs.


## Optional joint research controls

The user-owned integration branch adds two environment-driven controls in `backend/enhancement/stages.py`. `LUMENOIR_RETINEX_DOWNSAMPLE` is validated as an integer from 1 to 4 and passed into both `retinex.enhance` and `torch_retinex.enhance_tensor`. The baseline default is 2. `LUMENOIR_HUELOCK_SOURCE_ANCHOR` is validated in [0, 0.55] and passed, with the original RGB image, into both CPU and tensor HueLock. Its default is 0, which preserves the previous output path. Source anchor diagnostics are returned by the existing HueLock stage diagnostics when capture is enabled.
