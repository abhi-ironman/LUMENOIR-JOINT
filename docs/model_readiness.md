# Zero-DCE++ readiness

## Verified locally

- Checkpoint: `backend/weights/zero_dce_plus_plus.pth`
- Size: 52,395 bytes
- SHA-256: `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84`
- Strict load: succeeds (`strict=True`; 10,561 parameters, 28 tensors).
- CPU inference: succeeds; a 64×64 RGB image returns a 64×64 RGB image.
- Deep-learning mode: succeeds on a 96×64 RGB image; output is 96×64 RGB.
- Hybrid mode: succeeds on a 96×64 RGB image; output is 96×64 RGB.
- Colour path: RGB `[0, 1]` input, clipped `[0, 1]` output, encoded as 8-bit
  RGB; dimensions are restored exactly after padding/cropping.

A registration bug was found and fixed during this audit: importing the Kaggle
runner alone did not import `backend.deep_learning`, so the model registry was
empty. The runner now imports the model package.

## Device status

- Local environment: torch 2.15.0, CUDA unavailable, Apple MPS available.
- User-run local verification on 7 November 2026: `/api/enhance/status` reported
  `state: ready`, `model_id: zero_dce_plus_plus`, resolved model device `mps`,
  and `ready: true`. The browser workspace then returned all expected
  intermediate artifacts and the final output for an uploaded night-street
  photograph. This confirms a functional local smoke test; it is not a
  standardized quality benchmark.
- `cuda` compatibility cannot be executed locally because there is no CUDA
  device. The code path is present in `backend/inference/device.py`:
  `cuda`, `cuda:N`, and `auto` resolve to CUDA when available.
- On Kaggle GPU, `device: "auto"` is expected to resolve to `cuda:0`. This must
  be confirmed in the run’s `device.json` / `run_config.json`.
- Explicit `cuda` requests fail loudly rather than silently falling back to CPU.

## Checkpoint transfer

The checkpoint is not inside the permanent Kaggle low-light dataset. Transfer
it separately and either:

- place it at `/kaggle/working/zero_dce_plus_plus.pth`, matching both prepared
  configs, or
- edit the `weights` field in the configuration.

No weight download or modification is performed by the repository.

## Important limits

- The evaluator does not resize images for inference. SICE at 5997×3999 would
  be passed at native size in a model mode; use a separate SICE protocol and
  confirm memory before hybrid/deep-learning SICE runs.
- CUDA correctness and GPU memory behavior remain unverified locally.
