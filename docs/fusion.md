# LumenFuse

`backend/fusion/lumen_fuse.py` implements the noise-aware, Laplacian-pyramid
fusion from `final_decision.md` §4. It is wired into the API pipeline as the
`fusion` stage.

## Inputs

Three float32 RGB candidates in [0, 1] with identical dimensions:

1. `I0` — the original low-light image;
2. `C` — the LumenRetinex output (pre-denoise);
3. `D` — the Zero-DCE++ output (pre-denoise).

LumenScan information is required. If `scan_result` is omitted, `scan(I0)` is
computed; it supplies `sigma`, the full-resolution `noise_map` (`σ_loc`), and
the fusion exponents/`tau`.

## Weights

Per pixel and candidate `k`:

```
W_k = (Contrast_k + ε)^wc · (Sat_k + ε)^ws · (Expo_k + ε)^we · Noise_k^wn
```

- `Contrast = |Laplacian(GaussianBlur(Y_k, σ=1), ksize=3)|`
- `Sat = std(R_k, G_k, B_k)` across the three channels
- `Expo = exp(−(Y_k − 0.5)² / (2·0.2²))`
- `Noise = 1 / (1 + (Ĝ_k·σ_loc/τ)²)`
- `Ĝ_k = clip(blur(Y_k,5)/(blur(Y_0,5)+0.02), 1, 30)`

`(wc, ws, we, wn, τ)` come from LumenScan's fusion bucket. The original
candidate is multiplied by 3 where `max_c I0 > 0.85` (highlight protection).

Normalization follows the specification exactly:

```
Ŵ_k = (W_k + 1e-6) / Σ_j (W_j + 1e-6)
```

Each map is then refined with `guided_filter(guide=Y0, r=8, eps=1e-2)`,
clipped to non-negative values and renormalized. If a pixel's refined weights
sum to zero, it falls back to uniform `1/3` weights.

## Pyramid fusion

```
L = min(5, floor(log2(min(H, W))) - 3)
```

`L` is the top pyramid index, so `L + 1` levels are used. Weight maps are
Gaussian pyramids; candidates are Laplacian pyramids; each level is combined as
`Σ_k GaussianPyr_l(W_k) · LaplacianPyr_l(candidate_k)` and collapsed with
explicit destination sizes. The result is clipped to [0, 1] and has the exact
input dimensions. OpenCV's `createMergeMertens` is not used.

## Usage

```python
from backend.classical_cv.scan import scan
from backend.fusion.lumen_fuse import fuse

scan_result = scan(original)
result = fuse(original, classical, deep_learning, scan_result=scan_result)
result.image, result.weights, result.levels

# Explicit backends (the pipeline uses "auto")
cpu = fuse(original, classical, deep_learning, backend="cpu")
gpu = fuse(
    original,
    classical,
    deep_learning,
    backend="torch",
    device="cuda:0",
    precision="float32",
)
```

`compute_weights` is public for diagnostics; `pyramid_levels`,
`gaussian_pyramid`, `laplacian_pyramid`, `collapse_pyramid` and
`normalize_weights` are testable helpers.

## Backends
## Experimental DL-primary hybrid

The original LumenFuse remains the default. The optional
`LUMENOIR_HYBRID_STRATEGY=dl_primary` path starts from the Zero-DCE++ candidate
and transfers only a bounded high-frequency luminance residual from the
classical candidate. It uses three evidence terms: classical detail must be
closer to the observed source detail than the DL detail, the source must contain
measurable local detail, and the LumenScan local-noise estimate must be below
the configured limit. The correction is bounded before being converted to a
small RGB scale, so the DL candidate remains the base image.

This is an experimental heuristic, not a measured quality improvement. It is
disabled by default and does not replace the legacy LumenFuse implementation.
The evaluation matrix can select it explicitly:

```bash
python -m training.kaggle_eval.compare_checkpoints \
  --config /path/to/eval_config.json \
  --checkpoint finetuned=/path/to/best.pth \
  --datasets lol_v2_real,lol_v2_syn \
  --hybrid-strategy dl_primary \
  --retinex-downsample 2 --hue-anchor 0 \
  --device cuda \
  --output-dir /path/to/dl-primary-evaluation
```

Use a validation split to assess this strategy before any blind test. Do not
change the default to `dl_primary` unless the same held-out images and metric
configuration show a repeatable benefit over standalone deep learning.


`backend/fusion/lumen_fuse.py` owns the algorithm: candidate validation, the
four weight factors, highlight protection, normalization, guided refinement,
Gaussian/Laplacian pyramids, level blending and reconstruction. Backend-specific
numeric primitives are separated behind a small ops interface:

| Backend | Module | Primitives |
| --- | --- | --- |
| `cpu` | `lumen_fuse.py` (`_NumpyOps`) | The existing `cv2.GaussianBlur`, `cv2.Laplacian`, `cv2.blur`, `box_filter`, `cv2.pyrDown`, `cv2.pyrUp` calls |
| `torch` | `torch_backend.py` (`TorchFusionOps`) | CUDA/CPU/MPS tensors, `conv2d`, reflect padding, torch elementwise ops |

`fuse(..., backend=...)` accepts:

- `"cpu"` — default; the original OpenCV path. Works without PyTorch.
- `"torch"` — explicit torch backend. `device` may be `"cuda"`, `"cuda:N"`,
  `"cpu"` or `"mps"`; `precision` may be `"float32"` or `"float16"`.
- `"auto"` — uses CUDA when available and the CPU path otherwise. GPU
  initialization failures fall back to the CPU path with a warning.

The hybrid pipeline stage requests `"auto"`, so CPU-only deployments keep the
original path and CUDA deployments accelerate fusion automatically. The
optional overrides are `LUMENOIR_FUSE_BACKEND`, `LUMENOIR_FUSE_DEVICE` and
`LUMENOIR_FUSE_PRECISION`.

### GPU requirements

The torch backend requires the optional inference stack (`torch>=2.4`) and a
working CUDA/MPS device. `auto` only selects CUDA; MPS is available only when
requested explicitly (`backend="torch", device="mps"`). The CPU backend has no
extra requirements beyond the existing NumPy/OpenCV stack.

### Numerical equivalence

The torch backend mirrors OpenCV's borders (reflect vs reflect-101), derives the
Gaussian/Laplacian kernels from the installed OpenCV build, and uses the
standard 5-tap pyramid kernels. On the local validation set (textured, dark,
noisy, colour-cast, all-black, highlighted, odd-sized images and sizes from
17×19 to 640×640):

- CPU vs torch FP32 image MAE ≤ 1.3e-6, max absolute difference ≤ 7.9e-5,
  PSNR ≥ 103 dB, SSIM = 1.0.
- Refined weights: MAE ≤ 5e-7, max absolute difference ≤ 7e-6.
- A sharp highlight edge is the hardest case: image MAE ≈ 6e-8, max difference
  ≈ 7e-7. The diagnostic pre-refinement weights can differ by ≈ 1e-3 at
  individual flat pixels because the weight exponents amplify tiny contrast
  differences.
- FP16 is numerically safe on the measured images (MAE ≤ 5.6e-4, max difference
  ≤ 2.8e-3, PSNR ≥ 62 dB, SSIM = 1.0), but remains opt-in; production defaults
  to FP32.

These are measured local results, not Kaggle T4 results. See
[`fusion_benchmarks.md`](fusion_benchmarks.md) for the benchmark methodology and
the exact CUDA command.

## Tests

`tests/test_lumen_fuse.py` covers the original CPU behaviour: normalization and
zero-weight fallback, identical-candidate reconstruction, well-exposed
preference, dark-region enhancement, noise penalty, non-uniform regional
selection, highlight protection, colour-distorted candidates, extremely
dark/noisy and all-black stability, halo rate at a strong edge, odd dimensions,
pyramid round-trip and level count, invalid candidates/configurations and
mismatched LumenScan data. `tests/test_lumen_fuse_torch.py` adds CPU-vs-torch
parity, odd and tiny dimensions, dark/noisy/highlight/colour-cast inputs, weight
parity, CUDA-unavailable and initialization-failure fallbacks, FP16 checks and
the pipeline environment selector.

## Limitations

- CPU/GPU parity was measured on synthetic and local LOL-v2 images, not on a
  large held-out photograph corpus. The output targets (MAE ≤ 1e-3, SSIM ≥
  0.999) are met on the measured set; a sharp highlight edge reaches ~1.8e-3
  maximum difference at a few boundary pixels while keeping image MAE ≈ 1e-4.
- No Kaggle T4 measurement has been performed in this environment. Local
  measurements use Apple MPS and must not be presented as T4 results.
- The torch backend includes CPU-to-GPU input copies and a GPU-to-CPU output
  copy each call. On small images that overhead can make it slower than OpenCV;
  `auto` is intended for CUDA and falls back to CPU when CUDA is absent.
- FP16 is opt-in. It stayed within the measured quality targets, but FP32 is the
  default because FP16 quality on a wider image set has not been established.
- MPS peak memory is reported as the post-call allocated amount; CUDA uses
  `max_memory_allocated`. CPU memory is not reported by the benchmark.
- `blur(...,5)` is implemented as a 5×5 kernel, matching EdgeGuard's
  interpretation; the spec does not distinguish kernel size from radius.
- `L` is interpreted as the top pyramid index (L+1 levels), matching the
  pseudocode's `0..L` loop.
- In very flat regions the contrast and saturation factors are near zero, so the
  `+1e-6` normalization term makes the weights close to uniform. This follows
  the specified formula, but it means exposure differences have little effect in
  perfectly smooth areas.
- Weight exponents and `tau` are the untuned LumenScan starting values.
- The noise term penalises candidates by effective gain, not by measured
  candidate noise; a clean and a noisy candidate with the same gain receive the
  same penalty (as specified).
- The API pipeline applies EdgeGuard, HueLock and the adaptive alpha blend after
  fusion. The optional closed-loop passes are not implemented.
