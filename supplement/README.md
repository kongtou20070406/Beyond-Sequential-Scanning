# Supplementary experiments

This directory supports supplementary appendices G and H.
It keeps component training, protocol evaluation, and fixed-image timing separate.
The current main and component GoPro results use native full-image inference.
The old tiled files remain historical evidence; they are not current paper targets.

## Check the files and collect recorded results

Run these commands from the package root:

```text
python paper.py verify
python tools/check_release.py --load
python supplement/verify.py
python paper.py collect --suite gopro-components --recorded --output runs/recorded-components
python paper.py collect --suite gopro-native --recorded --output runs/recorded-native
python paper.py collect --suite spatial-mamba --recorded --output runs/recorded-spatial
```

The `--recorded` command reads the included metrics. It does not evaluate a model.
The load check verifies weight keys and tensor shapes. It does not reproduce test metrics.
See `THIRD_PARTY.md` for included licenses and unresolved upstream license coverage.
SS2D and Spatial-Mamba need their official extensions for import.
The CPU load check does not run their CUDA forward operation.

## Evaluate the component endpoints

Prepare GoPro with the root README command.
The data can use the flat `test/blur` and `test/sharp` folders from that command.
The evaluator also accepts the official sequence layout.

```text
python paper.py eval --suite gopro-components --data-root data --output runs/components
python paper.py eval --suite gopro-native --data-root data --output runs/native
```

All GoPro evaluations use FP32 and disable TF32.
Component metrics use one native full-image forward.
The historical tiled protocol uses core 384, halo 64, and overlap 64.
Native metrics use one full-image forward with no extra outer padding.
The original carrier boundary behavior stays active.
Each condition uses all 1111 test frames. Each frame has equal weight.
Do not combine native metrics with tiled metrics.

The formal seed 42 checkpoint is the primary component reference.
The separate full repeat is historical evidence and is outside the current registry.
Keep both checkpoints and both result records.
Do not insert the new full run into the main seeds 42, 43, and 44 mean.
An absolute difference below 0.1 dB is a descriptive near-parity convention.
It is not a statistical equivalence test or a pre-registered threshold.

## Train a new component endpoint

```text
python paper.py train --suite gopro-components --data-root data --output runs/new-components
python paper.py eval --suite gopro-components --trained --data-root data --output runs/new-components
```

Supplementary GoPro training uses seed 42, 49350 updates, BF16, effective batch 32,
microbatch 8, and crop 128. The shared random initial state and training image order
are included in `reference`. A new run does not load a trained endpoint.
The state-dynamic run uses activation recomputation.
Half delay retains the graph for previous sources and adds interpolation and storage.
It is a partial timing intervention. Its arithmetic and memory traffic are not matched.

The original new full run had numerical failures and resumed.
The state-dynamic run continued from Linux on Windows after update 33079.
The optimizer, scheduler, and main random states were restored.
Data-loader worker random states restarted. Environments differed across some conditions.
A new uninterrupted run does not replay these failure and continuation trajectories.
Use `python supplement/run.py --help` for an explicit resume of a new supplementary run.
Do not treat a resumed run as another independent seed.

## Build and evaluate Spatial-Mamba

The carrier has 22 official StructureAwareSSM slots with state dimension 1 and expand 2.
It retains the VIVNet residual and feed-forward structure.
It is not the full classification model. Parameter counts are not forced to match.
The adapter uses unchanged official model Python from the commit in
`third_party/SpatialMamba/SOURCE.json`. The license and required official sources are included.
The `build_source` tree contains the recorded CUDA 13 compatibility changes.
The patch is a review record of those changes. Build from `build_source` through `setup.py`.

From `supplement/third_party/SpatialMamba`, run:

```text
python -m pip install --no-build-isolation .
python -c "import selective_scan_cuda_oflex_rh, dwconv2d; print('PASS')"
```

The default build target is compute capability 12.0, as used by the RTX 5090 run.
Set `TORCH_CUDA_ARCH_LIST` for another supported device.
The original Windows build used MSVC, PyTorch 2.10.0, and CUDA 13.0.
The portable build selects C++ flags for Windows or Linux.
The Linux branch was not tested in this release check.
Do not replace an unavailable extension with another scan or depthwise operator.

Then run from the package root:

```text
python paper.py eval --suite spatial-mamba --data-root data --output runs/spatial
python paper.py train --suite spatial-mamba --data-root data --output runs/new-spatial
```

The fixed endpoint is update 55800, seed 42, batch 2, crop 256, and FP32 training.
Evaluation uses O-HAZE IDs 41–45, long side 512, bicubic antialias resize,
RGB PSNR, and direct RGB SSIM with no image pre-pooling.
SSIM uses an 11 by 11 Gaussian window, sigma 1.5, K values 0.01 and 0.03,
and data range 1. The prediction is clamped to [0, 1]. The resized target is unchanged.
The five test images are not independent training repeats.
The old ours and SS2D values in the comparison are the existing rounded paper values.
`evidence/ohaze_ef` preserves the existing EF per-image result and its checkpoint hash.
It also records the training-log times and the unresolved cause of an earlier metric difference.

## FP32 inference and timing

The full radial model uses joint input heads and direct five-step propagation
in the restoration evaluation commands. Use [benchmark/README.md](benchmark/README.md)
for numerical verification, paired timing, and full-test evaluation commands.
The selected implementation and SS2D retain all three original timing processes.
Obsolete optimization entry points are archived outside this package.

## CSPN generator control

```text
python supplement/run.py --id ohaze-cspn-context-seed42 --action eval --data-root data/ohaze --output runs/cspn-context
python supplement/run.py --id ohaze-cspn-context-seed42 --action train --data-root data/ohaze --output runs/cspn-context-training
```

The first command evaluates the released endpoint. The second starts a new
training run. Training loads the released initial weights and uses the recorded
sample-key order and paired transforms. Its scope is one random seed. Existing
CSPN-style and ours endpoints are reused in the recorded comparison.

## Direct O-HAZE SSIM records

`evidence/ssim_reassessment_20260912` contains the direct SSIM reassessment.
It includes 38 groups with five image scores each and 160 same-size scale cells.
Each record includes measured PSNR, direct SSIM, the former pooled SSIM,
and the checkpoint SHA256. See its `protocol.json` for the metric settings.

The experiment registry uses the new SSIM values for 23 current paper conditions.
It retains the published PSNR values. The five other historical O-HAZE conditions
were not measured again. They have no current `expected.ssim` value.
Their old scores remain under `legacy_pooled_ssim`.
The old result files remain available. Spatial-Mamba and the CSPN context control
now point to the new direct SSIM result files for recorded-result collection.
The files under `results/ohaze-spatial-mamba-seed42` and
`results/ohaze-cspn-context-seed42` are historical pooled-SSIM evidence only.
The active collector does not use them as current SSIM results.
GoPro values are unchanged.

## Resource and source records

`evidence/INFERENCE_RESOURCE_SUMMARY.json` holds the recorded resource statistics.
`evidence/INFERENCE_MACS_REFERENCE.json` holds the static 256-square MAC references.
The per-image files preserve available timing and memory observations.
Native seed 42 quality covers 1111 frames; its new resource records cover 1078 frames.
The earlier 33 metrics per method were reused without rerunning them.
Historical missing resource fields remain missing.
Half-delay evaluation used two shards of 556 and 555 frames.
Its evaluation wall time is not a single-GPU latency.

CUDA-synchronized forward wall time excludes decode, transfer, and quality metrics.
For tiled evaluation it includes tile assembly and output clamp.
For native evaluation the original record ends timing before clamp.
Memory values are process allocated/reserved peaks in MiB, not whole-device usage.
First-call overhead and shared-host effects remain in the records.

Runtime overlays contain only changed files. `runtime.py` reconstructs the recorded
runtime in the selected output directory and verifies its source hashes.
`SOURCE_PROVENANCE.json` links exported files to the hashes of the archived sources.
Published JSON removes machine paths and operational identifiers.
File hashes can therefore differ while metrics remain unchanged.
No dataset images, access credentials, personal machine binaries, or scheduler scripts are included.
