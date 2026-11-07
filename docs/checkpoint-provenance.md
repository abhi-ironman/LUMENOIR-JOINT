# Checkpoint provenance

This document records the two model files recovered from the previous
`My Drive/LUMENOIR/checkpoints` archive. Binary weights are not committed to
Git; keep the files in a local/Kaggle weight directory.

## Published pretrained baseline

- File in archive: `zero_dce_plus_plus_Epoch99.pth`
- Size: 52,395 bytes
- SHA-256: `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84`
- Git blob SHA-1: `3d1a7b8f5aee769d8ae55531cd43c4cf8d45efa8`
- 28 tensors / 10,561 parameters
- This is the published Zero-DCE++ Epoch99 baseline already documented by
  `backend/deep_learning/zero_dce_plus_plus.py`.

## Archived legacy LUMENOIR checkpoint

- File in archive: `zero_dce_plus_plus.pth`
- Safe comparison filename: `zero_dce_plus_plus_legacy_epoch100.pth`
- Size: 163,337 bytes
- SHA-256: `90997def427e5e81c2c9d32eed9781e6cb39040ac2c897e54ab0cd1706000662`
- Git blob SHA-1: `66afca7d4164e7bdb9637f0f639b822283d8a8f7`
- Payload: nested `state_dict`, epoch 100, reported loss 0.7292072025478863,
  and optimizer state
- Training config recorded in the checkpoint: `data/train`, 100 epochs,
  batch size 8, 256-pixel patches, learning rate 1e-4, weight decay 1e-4,
  seed 42, Zero-DCE++ official Epoch99 initialization
- All 28 tensor keys and shapes match the published Epoch99 state dict. All
  tensors are finite, and every tensor differs numerically from the baseline.

The archive does not contain training history, validation metrics, or a
complete data manifest. The prior Colab notebook shows an intended LOL-v1
training command, but the archived checkpoint alone cannot prove the exact
materialized training files or establish held-out performance.

The current LUMENOIR-JOINT inference loader accepts nested `state_dict`
payloads, verifies the 10,561 parameter count, 28 tensors and architecture
keys, and loads strictly. Therefore this checkpoint is suitable as an extra
legacy comparison candidate without replacing the published pretrained file.

## Research handling

Use distinct labels:

- `pretrained`: official Epoch99 baseline
- `legacy_epoch100`: recovered previous LUMENOIR trained checkpoint
- `abhi`: new `best.pth` from the current LUMENOIR-JOINT fine-tuning recipe

Do not call `legacy_epoch100` the output of the new SICE+LOL recipe, and do
not claim its quality metrics without running it through the same evaluation
runner and dataset splits as the other candidates.
