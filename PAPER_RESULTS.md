# Paper experiment map

Use `paper.py list` to show every experiment ID.
Use `paper.py eval --suite NAME` to evaluate a complete table.
Use `paper.py collect --suite NAME` to compare the results with the paper.

## Selection rules

| Dataset | Checkpoint rule | Evaluation rule |
|---|---|---|
| O-HAZE | Update 55,800; seed 42 | IDs 41 to 45; FP32; long side 512; RGB PSNR; direct RGB SSIM |
| GoPro | Update 49,350; seeds 42, 43, 44 | 1,111 test pairs; one native full-image FP32 call; TF32 disabled; no extra outer padding |
| Haze4K | Update 100,000; seed 42 | 1,000 test pairs; native whole-image FP32; pad to a multiple of four and crop before metrics; no test-time augmentation |

O-HAZE SSIM uses the full long-side-512 output without image pre-pooling.
The SSIM window is 11 by 11 with sigma 1.5 and K values 0.01 and 0.03.
The main GoPro comparison and component table use one native full-image call.
All frames use FP32 with TF32 disabled and no extra outer padding.

## Independent-training experiments

| Experiment ID | PSNR or accuracy | SSIM or Top-5 |
|---|---:|---:|
| `ohaze-native-seed42` | 25.8096 | 0.904283941 |
| `ohaze-dwconv-seed42` | 25.9001 | 0.901542139 |
| `ohaze-mdta-seed42` | 25.8712 | 0.903529525 |
| `ohaze-swin-seed42` | 25.5671 | 0.897180140 |
| `ohaze-ss2d-seed42` | 25.625 | 0.897918415 |
| `ohaze-cspn_style-seed42` | 26.2017 | 0.903972137 |
| `ohaze-ours-seed42` | 26.7755 | 0.906549466 |
| `ohaze-mair-seed42` | 22.8164 | 0.764535916 |
| `ohaze-matched_feedforward-seed42` | 26.4958 | 0.903904355 |
| `ohaze-step_conditioned_rules-seed42` | 26.6525 | Not measured; historical pooled value only |
| `ohaze-first_step_drive-seed42` | 26.5319 | 0.911714315 |
| `ohaze-identity_retention-seed42` | 26.4429 | 0.901647079 |
| `ohaze-terminal_readout-seed42` | 26.3239 | 0.904631329 |
| `ohaze-identity_retention_terminal_readout-seed42` | 26.2586 | 0.903925204 |
| `ohaze-static_rules-seed42` | 26.0944 | 0.903942657 |
| `ohaze-equal_parameter_real-seed42` | 26.4999 | Not measured; historical pooled value only |
| `ohaze-no_row_mass_projection-seed42` | 24.8043 | 0.893189096 |
| `ohaze-alternative_generator-seed42` | 25.9505 | Not measured; historical pooled value only |
| `ohaze-conjugate_symmetric-seed42` | 26.487 | Not measured; historical pooled value only |
| `ohaze-depth_k1-seed42` | 25.8834 | 0.901863146 |
| `ohaze-depth_k3-seed42` | 26.4699 | 0.904554653 |
| `ohaze-depth_k7-seed42` | 26.35 | 0.906914937 |
| `ohaze-p0t_diagonal_delay-seed42` | 26.4166 | Not measured; historical pooled value only |
| `ohaze-p0s_state_dynamic-seed42` | 26.3328 | 0.904061520 |
| `gopro-native-seed42` | See three-seed statistics | - |
| `gopro-native-seed43` | See three-seed statistics | - |
| `gopro-native-seed44` | See three-seed statistics | - |
| `gopro-cspn_style-seed42` | See three-seed statistics | - |
| `gopro-cspn_style-seed43` | See three-seed statistics | - |
| `gopro-cspn_style-seed44` | See three-seed statistics | - |
| `gopro-ss2d-seed42` | See three-seed statistics | - |
| `gopro-ss2d-seed43` | See three-seed statistics | - |
| `gopro-ss2d-seed44` | See three-seed statistics | - |
| `gopro-ours-seed42` | See three-seed statistics | - |
| `gopro-ours-seed43` | See three-seed statistics | - |
| `gopro-ours-seed44` | See three-seed statistics | - |

## GoPro three-seed statistics

The values below are mean plus or minus sample standard deviation.

| Mixer | PSNR | SSIM |
|---|---:|---:|
| native | 29.9024 +/- 0.1205 | 0.896065 +/- 0.002584 |
| cspn_style | 30.2023 +/- 0.0910 | 0.901086 +/- 0.001702 |
| ss2d | 30.3674 +/- 0.0685 | 0.904607 +/- 0.001215 |
| ours | 30.8687 +/- 0.0482 | 0.912960 +/- 0.000645 |

## MaIR checkpoint selection

The primary result uses the fixed update 55,800 checkpoint.
The best saved checkpoint is at update 30,000.
Its PSNR is 23.1019 dB. Its historical pooled SSIM is 0.87444.
No direct SSIM measurement is bundled for this checkpoint, so this SSIM is not
a current submission result. This is a separate checkpoint-selection result.
It is not a second independent training run.

```text
python reproduce.py --task restoration --dataset ohaze --mixer mair --action eval --data-root data/ohaze --checkpoint weights/ohaze/mair/seed42/best.pth --run-dir runs/mair-best
```

## Evidence limits

P0-T delays diagonal source values from the first update.
The fixed-weight timing test keeps the first update unchanged.
Do not compare the two deficits as an adaptation estimate.
P0-S has the same parameter count as the full model.
It has a higher MAC count because it generates a new kernel at each update.

Fixed-weight diagnostics have separate commands in `DIAGNOSTICS.md`.
This release covers the restoration experiments in the September 25 submission.
The experiment list separates propagation controls from main mixer comparisons.

## Supplementary component endpoints

These seed 42 results use all 1,111 GoPro test frames with one native full-image
call. The formal full model is the reference for every component comparison.

| Condition | PSNR (dB) | SSIM |
|---|---:|---:|
| Full | 30.8917521 | 0.913199417 |
| Identity retention | 30.9025066 | 0.913282027 |
| Terminal readout | 30.9126443 | 0.913168516 |
| State dynamic | 30.9480481 | 0.914123567 |
| Half delay | 30.8969842 | 0.913193091 |
| No normalization | 28.6148293 | 0.863856086 |

The no-normalization row includes its original endpoint and an evaluation-only
configuration. It does not provide a new training recipe. Its per-image and
per-video evidence is in `supplement/results/native_main`.

The full, identity-retention, terminal-readout, and half-delay MAC references
are 61.431349248 G at a 256-square input. The state-dynamic reference is
76.497813504 G. All five new conditions have 5,807,236 parameters.
Half-delay interpolation operations and history storage are additional costs.

The O-HAZE Spatial-Mamba matched-slot endpoint gives 22.7317753 dB and
0.788609362 direct SSIM over the five test images. It uses the official operator
with 22 carrier slots. It is not the full Spatial-Mamba classification model.

## Native-image evaluation and fixed-image timing

The `gopro-native` suite evaluates existing main checkpoints with one full-image
forward and no extra outer padding. These metrics agree with the current main table.
Use `python paper.py collect --suite gopro-native --recorded` for the included
per-seed values. Native evaluation adds no independent training run.

The default joint FP32 implementation has native fixed-image CUDA medians of
0.4935363 to 0.4961464 s across three processes. Same-process SS2D medians are
0.5126948 to 0.5144204 s, a latency reduction of 3.41% to 3.74%.
These are not dataset-wide latency averages. The 55-image native numerical panel
passed, with a maximum PSNR difference of 2.3841858e-6 dB from the reference.
See `supplement/evidence/default_native_20260913/SUMMARY.json`.
Historical tiled measurements remain in `supplement/benchmark/CURRENT_SUMMARY.json`.

See `supplement/README.md` for the exact entry points and evidence limits.

## Native-image three-seed summary, 2026-09-09

Each seed uses 1,111 images with one native-image forward.
Values after ± are sample standard deviations over three training seeds.

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| ours | 30.8686906 ± 0.0481710 | 0.912959649 ± 0.000645309 |
| ss2d | 30.3673797 ± 0.0684997 | 0.904607410 ± 0.001214662 |

This native protocol is now the main paper protocol. The original tiled results
remain historical records and are not current main targets.
The native resource records and their coverage limits are in
`supplement/evidence/NATIVE_THREE_SEED.json`.

## CSPN generator ablation

CSPN-style results belong to the `cspn-propagation` suite. The new O-HAZE
context-generator condition has 2,230,624 parameters. It uses the same hidden
generator structure as ours, with a head for eight real affinity coefficients.

| Method | PSNR (dB) | SSIM |
|---|---:|---:|
| CSPN-style, local generator | 26.2013695 | 0.903972137 |
| CSPN-style, context generator | 23.9342587 | 0.784573793 |
| Ours | 26.7752247 | 0.906549466 |

These values come from one evaluation batch. Existing reference weights were
reused. Only the context-generator condition was newly trained. The reference
replays differ from the original rounded table by less than 0.001 dB.

## Haze4K final endpoints

All rows use seed 42, 100,000 updates, and all 1,000 native test images. SSIM uses the direct RGB protocol. Original evaluation and checkpoint hashes are linked in each bundled FINAL.json.

| Configuration | PSNR (dB) | Direct SSIM |
|---|---:|---:|
| NSI | 30.1147020 | 0.982611619 |
| SS2D | 28.7409662 | 0.972696708 |
| Identity retention + final-state readout | 29.9225734 | 0.982131809 |
| Final-state readout | 29.3348917 | 0.980951566 |
| Identity retention | 30.0479113 | 0.982763997 |
| Weighted-state readout | 29.8398473 | 0.982006355 |
| Learnable propagation scaling | 30.3026418 | 0.983199844 |
| CSPN-style: replaced generator | 27.6520151 | 0.974878705 |
| VIVNet native | 29.8087766 | 0.980831946 |
| CSPN-style | 27.2493016 | 0.973602711 |
| DWConv | 29.6030767 | 0.981382203 |
| Swin | 28.0331918 | 0.975554870 |
| Without propagation gain control | 26.3747012 | 0.968841686 |
| Per-step kernel regeneration | 29.1378222 | 0.979345456 |

## GoPro joint ablation

Identity retention plus final-state readout, seed 42, 49,350 updates:
PSNR 30.8772027053 dB and direct SSIM 0.912575450 on all 1,111 native images.
Against the historical full seed-42 endpoint 30.8917520864 dB, the unrounded
PSNR difference is -0.0145493811 dB (-0.0145 in the paper).
This is a joint component replacement, not a repeat of the full model.
Use the gopro-joint-seed42 registry entry for its configuration and weights.
