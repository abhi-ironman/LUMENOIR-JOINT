# LUMENOIR-JOINT — Experimental results and close-out status

**Snapshot:** 7 November 2026  
**Branch:** `integration/lumenoir-research`  
**PR:** #1 (open, draft); `main` remains unchanged.

## Executive conclusion

LUMENOIR-JOINT is a functioning integrated low-light enhancement application with Classical CV, Deep Learning and Hybrid modes. However, the available evaluations do **not** demonstrate that the current Hybrid output is better than its strongest individual branch.

The preliminary fine-tuned Zero-DCE++ branch improves on the published pretrained checkpoint on LOL-v2 Synthetic across all four reference metrics. On LOL-v2 Real, results are mixed. The legacy LumenFuse Hybrid is worse than the fine-tuned Deep Learning branch on all four aggregate metrics on LOL-v1, LOL-v2 Real and LOL-v2 Synthetic.

**Recommended positioning:** present this as an integrated application combining classical and deep-learning approaches, not as proof that the fused model beats both individual systems. The fine-tuned Deep Learning mode is the best-performing measured mode in the available experiments.

## Fine-tuning checkpoint

The current fine-tuned checkpoint is a preliminary LOL-v1-only run: 388 training pairs, 97 validation pairs, 6 epochs completed, best checkpoint at epoch 1, early stopping enabled. SHA-256:

`60447a8680ef60b4dcfe7c142f0f25cad2cc879df78836e7cbfdc88b7a123adf`

This is not the full SICE + LOL recipe.

## LOL-v1 official eval15 (15 pairs)

All runs completed without recorded failures. Higher PSNR/SSIM is better; lower LPIPS/Delta E 00 is better.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Delta E 00 ↓ |
|---|---:|---:|---:|---:|
| Original | 7.7733 | 0.1914 | 0.5596 | 37.0757 |
| Classical CV | 14.8016 | 0.3534 | 0.5785 | 15.9079 |
| Published pretrained Zero-DCE++ | 15.1654 | 0.6825 | 0.2966 | 19.3066 |
| Fine-tuned Zero-DCE++ | **18.3955** | **0.7116** | **0.2846** | **13.2305** |
| Pretrained Hybrid | 15.1673 | 0.4037 | 0.5477 | 16.0628 |
| Fine-tuned legacy LumenFuse Hybrid | 15.9147 | 0.4228 | 0.5278 | 14.7262 |

Fine-tuned DL beats the fine-tuned Hybrid on all four aggregate metrics. This 15-image test has been inspected during development and is diagnostic, not a fresh blind final test.

## LOL-v2 Real (100 pairs)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Delta E 00 ↓ |
|---|---:|---:|---:|---:|
| Classical CV | 16.3387 | 0.4082 | 0.5080 | 14.0094 |
| Published pretrained Zero-DCE++ | 19.0140 | **0.7248** | **0.2608** | 13.4607 |
| Fine-tuned Zero-DCE++ | **19.1418** | 0.7121 | 0.2735 | **12.3303** |
| Fine-tuned legacy LumenFuse Hybrid | 17.5506 | 0.4805 | 0.4549 | 12.8317 |

Fine-tuning slightly improves PSNR and colour error, but pretrained Zero-DCE++ retains better SSIM and LPIPS. This is mixed, not an overall win.

## LOL-v2 Synthetic (100 pairs)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Delta E 00 ↓ |
|---|---:|---:|---:|---:|
| Classical CV | 16.0978 | 0.6548 | 0.3221 | 14.5795 |
| Published pretrained Zero-DCE++ | 17.4315 | 0.8020 | 0.2249 | 16.3906 |
| Fine-tuned Zero-DCE++ | **18.7041** | **0.8272** | **0.1983** | **13.2107** |
| Fine-tuned legacy LumenFuse Hybrid | 16.9830 | 0.7175 | 0.2832 | 13.9023 |

The fine-tuned DL branch improves on the published pretrained checkpoint on all four metrics here.

## Hybrid versus fine-tuned Deep Learning

- LOL-v1 eval15: Hybrid is worse on all four aggregate metrics.
- LOL-v2 Real: Hybrid minus DL = PSNR −1.5912 dB, SSIM −0.2316, LPIPS +0.1814, Delta E 00 +0.5014.
- LOL-v2 Synthetic: Hybrid minus DL = PSNR −1.7211 dB, SSIM −0.1097, LPIPS +0.0849, Delta E 00 +0.6916.

A validation-only fixed pixel blend sweep on 97 LOL-v1 validation pairs also failed to beat standalone fine-tuned DL. A temporary DL-primary correction changed validation metrics only negligibly and was removed from the integration branch. No further fusion research is planned for this close-out.

## Previous user checkpoint and friend's model — comparison gaps

The recovered previous LUMENOIR checkpoint is architecture-compatible and contains epoch-100 metadata. SHA-256:

`90997def427e5e81c2c9d32eed9781e6cb39040ac2c897e54ab0cd1706000662`

It was **not** included in the LOL-v1/LOL-v2 comparison matrices above. We therefore cannot claim the current fine-tuned checkpoint beats the previous epoch-100 model.

The inspected upstream `Chiranthcommits25/Lumenoir` README documents the published pretrained Zero-DCE++ model and a training placeholder. No distinct friend-trained checkpoint was supplied for a matched evaluation. The published pretrained checkpoint in the tables is a baseline for the upstream project, not evidence about an additional private friend-trained model.


## Historical LUMENOIR results supplied after the joint evaluation

A second archive supplied on 7 November contains older results from the previous
LUMENOIR implementation. These results are useful historical evidence and
should not be conflated with the current LUMENOIR-JOINT pipeline.

### Previous LUMENOIR — LOL-v1 (15 images)

| Mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Delta E 00 ↓ |
|---|---:|---:|---:|---:|---:|
| Classical | 16.9764 | 0.7195 | 0.2632 | 5.8613 | 14.5029 |
| Deep Learning | 15.9771 | 0.7097 | 0.2880 | 5.9720 | 16.8721 |
| **Hybrid** | **17.8233** | **0.7467** | **0.2496** | 5.8870 | **13.2399** |

On this small set, the old Hybrid has the best mean PSNR, SSIM, LPIPS and
Delta E 00. Classical has a slightly lower (better) mean NIQE.

### Previous LUMENOIR — LOL-v2 Real (100 images)

| Mode | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Delta E 00 ↓ |
|---|---:|---:|---:|---:|---:|
| Classical | **19.3826** | 0.7444 | 0.2795 | 6.3304 | **12.2596** |
| Deep Learning | 14.2615 | 0.6820 | 0.3196 | 6.4104 | 21.3940 |
| **Hybrid** | 19.2160 | **0.7638** | **0.2612** | **6.3068** | 12.2876 |

The previous Hybrid is better than the previous Deep Learning mode on all five
aggregate metrics. Against previous Classical, it has higher SSIM and better
LPIPS/NIQE, while Classical is slightly ahead on PSNR and Delta E 00. Per-image
counts also favour the previous Hybrid over previous Deep Learning on 81/100
images for PSNR, 77/100 for SSIM, 88/100 for LPIPS, 66/100 for NIQE and 80/100
for Delta E 00.

### Relationship to the current joint evaluation

The 100 LOL-v2 Real filenames in the historical archive match the current
LUMENOIR-JOINT LOL-v2 Real manifest exactly (00690.png through 00789.png).
This makes the historical report especially useful: it suggests that the
previous LUMENOIR hybrid implementation/settings were materially stronger on
this named test set than the current joint Hybrid result.

It is **not** a controlled cross-version A/B result. The historical archive
does not include the source commit, checkpoint SHA-256, image/reference hashes
or complete metric/preprocessing configuration. Its CSVs show a classical
fusion base weight of 0.4 and DL strength scale 0.75; the final profile archive
also records Retinex illumination scale 0.33. The current joint comparison uses
a different integration pipeline/configuration and records checkpoint hashes.
The older neural checkpoint is not included in the archive.

Therefore, retain this as historical evidence of a stronger previous LUMENOIR
hybrid, and as a signal that the joint port/configuration may have regressed.
Do not merge the two sets of numbers into one leaderboard or claim they are
fully comparable until the old checkpoint and exact implementation/config are
recovered and re-evaluated under one harness.

### Friend output in the historical archive

The archive includes one side-by-side night-scene output from the friend's
model, alongside previous LUMENOIR Classical, Deep Learning and Hybrid images.
This is a single visual example without a ground-truth reference or paired
metric evaluation. It can support a qualitative demonstration only; it cannot
establish that either system is objectively better than the other.

## Final limitations and recommendation

- No full SICE + LOL fine-tuning run.
- No matched evaluation of the recovered epoch-100 user checkpoint.
- No separate friend-trained checkpoint was provided/evaluated.
- The LOL-v1 and LOL-v2 test sets have been inspected during development, so do not describe them as untouched blind final tests.
- No NVIDIA T4 deployment latency/memory benchmark.
- No full ExDark/DICM or 100-image do-no-harm gate.

Keep the three application modes available. Recommend the fine-tuned Deep Learning mode for best measured image quality; retain Classical CV as an independent mode and the legacy Hybrid as an experimental option. Do not claim the current Hybrid beats both standalone systems.

**Final evidence-supported claim:** LUMENOIR-JOINT is a functioning integrated low-light enhancement application. Its preliminary fine-tuned Zero-DCE++ mode improves on the published pretrained baseline on LOL-v2 Synthetic, is mixed on LOL-v2 Real, and the current hybrid fusion does not outperform the fine-tuned DL branch on the evaluated datasets.
