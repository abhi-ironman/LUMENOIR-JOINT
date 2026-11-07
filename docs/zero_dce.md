# Zero-DCE++ (NoirNet) inference

Zero-DCE++ is the selected deep-learning backbone (`final_decision.md` §2).
This document records the integration as it exists in the repository. It is
wired into the `deep_learning` pipeline stage through `ModelManager`; the hybrid
and deep-learning modes require a ready checkpoint.

## Provenance

| Item | Value |
| --- | --- |
| Repository | `https://github.com/Li-Chongyi/Zero-DCE_extension` |
| Architecture file | `Zero-DCE++/model.py` (branch `main`) |
| Checkpoint | `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth` |
| Local path | `backend/weights/zero_dce_plus_plus.pth` (git-ignored) |
| Size | 52,395 bytes |
| Git blob SHA | `3d1a7b8f5aee769d8ae55531cd43c4cf8d45efa8` |
| SHA-256 | `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84` |
| Parameters | 10,561 in 28 tensors, all float32 |
| Licence | CC BY-NC 4.0 (academic research use); cite Li, Guo, Loy, TPAMI 2021 |

Download (only when provisioning a deployment):

```bash
curl -L --fail --output backend/weights/zero_dce_plus_plus.pth \
  "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"
```

Verification performed on 6 November 2026:

- file size and git blob SHA match the GitHub contents API metadata exactly
- `torch.load(..., weights_only=True)` returns a plain `OrderedDict`
- key set is identical to the architecture (missing `[]`, unexpected `[]`)
- strict `load_state_dict` succeeds
- no original Zero-DCE key names are present; the original checkpoint has
  about 79K parameters, this one has 10,561

## Architecture

- Seven depthwise-separable blocks: 3×3 depthwise convolution followed by a 1×1
  pointwise convolution, 32 feature channels.
- Skip concatenations: `cat(x3, x4)`, `cat(x2, x5)`, `cat(x1, x6)`.
- Curve map `alpha = tanh(...)`, applied eight times as
  `x + alpha * (x² − x)`.
- Downsampling: bilinear, `align_corners=False`.
- Upsampling: `nn.UpsamplingBilinear2d`, equivalent to bilinear with
  `align_corners=True`. This is reproduced explicitly; using the PyTorch
  default (`False`) would change the output.
- No batch norm, no mean/std normalisation.

### Alpha convention

The authors' curve is `x + raw_alpha·(x² − x)`, which equals
`x + (−raw_alpha)·x·(1 − x)`. `final_decision.md` writes the curve as
`x + α·x·(1 − x)` and multiplies α by the DL strength `s`. The wrapper therefore
returns both:

- `raw_alpha`: the tanh output used by the official implementation
- `curve_alpha = -raw_alpha`: the sign-corrected coefficient retained for
  diagnostics. `DeepLearningStage` applies LumenScan's selected DL strength by
  interpolating the model output back toward the source image.

## Input, output and geometry

- Input: validated RGB `PIL.Image`, converted to float32 `/255`, NCHW, `[0,1]`.
- The image is padded bottom/right to a multiple of `scale_factor` with reflect
  padding (replicate for tiny images). The authors' script crops instead and
  discards up to `scale_factor − 1` border pixels; padding keeps the full frame.
- Output: float32 clipped to `[0,1]`, encoded as RGB at the **exact input size**.
- No resizing happens inside the model. The API currently validates the upload
  against its maximum inference pixel limit; it does not silently resize the
  input before inference. Keep test images within that limit.

## Device selection

`LUMENOIR_INFERENCE_DEVICE` accepts:

| Value | Behaviour |
| --- | --- |
| `cpu` (default) | Always available. |
| `mps` | Apple Silicon Metal; raises `device_unavailable` if unsupported. |
| `cuda` / `cuda:N` | CUDA; raises `device_unavailable` if unsupported or the index is out of range. |
| `auto` | MPS → CUDA → CPU. The only mode that falls back. |

The manager resolves the device once at load and reports the resolved string in
`/api/enhance/status`. An explicit device is never silently replaced.

### Measured on this machine

Apple Silicon, Python 3.14, torch 2.15.0, 1200×900 input, single image:

| Device | Time |
| --- | --- |
| CPU | 31.8 ms median (3 runs after warm-up) |
| MPS | 415 ms (second call); first MPS call includes ~1.5 s compilation |

These are single-machine observations, not a benchmark. The integration test
asserts only that CPU 1200×900 stays under the 10-second product target.

## Configuration

```bash
LUMENOIR_MODEL_ID=zero_dce_plus_plus
LUMENOIR_INFERENCE_DEVICE=cpu          # cpu | mps | cuda | cuda:N | auto
LUMENOIR_ZERODCE_SCALE_FACTOR=12       # 1..12; 12 matches the authors' test script
LUMENOIR_WEIGHTS_DIR=backend/weights   # default
```

`final_decision.md` mentions both scale 12 (§2.3) and scale 2–4 (§10). The
checkpoint has no scale-dependent parameters, so this is a runtime choice;
12 is the default for parity with the released inference protocol. 2–4 can be
benchmarked later without code changes.

## Dependencies

- `requirements-inference.txt`: `-r requirements.txt` + `torch>=2.4` + `numpy>=2.1`.
- `requirements.txt` (and the Dockerfile) are unchanged: the API starts without
  torch. If the model is selected without torch, the manager records
  `torch_missing` (503) and the rest of the API keeps working.
- No Python or PyTorch downgrade. The official README targets Python 3.7 /
  torch 1.0, but every operator used still exists in torch 2.15.

## Errors

| Code | Status | Cause |
| --- | --- | --- |
| `torch_missing` | 503 | Model selected but torch is not installed. |
| `device_unavailable` | 503 | Explicit device unavailable or invalid index. |
| `checkpoint_invalid` | 500 | Missing/unreadable checkpoint, wrong parameter count, or key mismatch. |
| `inference_failed` | 500 | Warm-up or forward pass failed on the selected device. |

The existing `weights_missing` / `model_not_available` behaviour is unchanged.

## Tests

- Unit: `tests/test_zero_dce_inference.py` (random checkpoints, no network).
- Integration: `tests/test_zero_dce_integration.py` (real checkpoint, marked
  `integration`, 120-second timeout, skipped when weights are absent).
- Import safety: a subprocess asserts `import backend.main` does not import
  torch or numpy.

```bash
python -m pytest -q
python -m pytest -m integration -o addopts="" tests/test_zero_dce_integration.py
```

## Fine-tuning (implemented)

The repository includes a device-aware Zero-DCE++ fine-tuning package at
`training/finetune/`. It preserves the inference architecture and exports a
strict-load-compatible `best.pth` plus a resumable `last_training.pt`.

- The configured LUMENOIR recipe uses Adam at `2e-5`, up to 30 epochs, early
  stopping, gradient clipping and the documented highlight/chromaticity loss
  terms.
- Training and validation manifests are separate; test records are not loaded
  by the trainer. Path overlap between train and validation aborts the run.
- The `best.pth` file is the inference artifact. `last_training.pt` is only for
  resuming training and must not be used as a model checkpoint.
- See [`training/finetune/README.md`](../training/finetune/README.md) for the
  Colab/Kaggle setup, dry-run, training, outputs and validation procedure.

**Dataset-scope note:** the legacy `zero_dce_plus_plus_lol.json` config
uses the committed 1,565 LOL train and 391 LOL validation pairs. The full
`zero_dce_plus_plus_full.json` config supports the planned SICE+LOL recipe:
SICE Part 1 is consumed as single-image zero-reference data, and paired LOL
validation records remain responsible for PSNR/SSIM/LPIPS checkpoint selection.
The split builder must first generate `fine_tune.json` and the mixed
`validation.json` from the real datasets. See
[`training/finetune/README.md`](../training/finetune/README.md) for the exact
data preparation and dry-run procedure.

The original Zero-DCE++ paper implementation used CUDA-specific helper code
and an old torchvision weights API; the repository's trainer uses the current
model implementation and the selected resolved device instead.

## Limitations

- The model is zero-reference; published LOL-v1 PSNR is around 14.7 dB, so it is
  a baseline rather than a high-PSNR method.
- CPU latency at the 4 MP inference ceiling has not been benchmarked.
- ONNX export (§9.6) is not implemented.
- Weights are never committed; deployments must provision the file.
