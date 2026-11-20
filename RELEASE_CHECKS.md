# Release checks

Original checks: 2026-09-13. Windows, Python 3.12, PyTorch 2.10, CUDA 13.0, and RTX 5090.
These release checks use existing weights. They start no training run.

## Current native default

The full radial model selects the fused CUDA FP32 path in its default forward.
Training, gradient-enabled calls, CPU, and autocast retain their model path.
The cache refreshes after depth parameters change. No state key changes.
The GoPro evaluator and benchmark default to a native full-image call.

The 55-image numerical panel passed with a maximum output difference of
1.2516976e-6 and a maximum PSNR difference of 2.3841858e-6 dB. A batch of two
rectangular inputs passed. A CPU backward pass after CUDA inference was finite.
Three independent native timing processes are included. See
`supplement/evidence/default_native_20260913` for raw records and limits.

All 17 current main/component GoPro conditions have 1,111-image reference records
linked to the released checkpoint hashes. The optimized path was checked on the
55-image panel; it was not re-evaluated on all three full test runs in this update.
See `supplement/evidence/NATIVE_MAIN_20260913.json`.

## Historical tiled FP32 checks

The restoration evaluation commands use the joint input heads and direct
five-step rollout for the full radial model. The inference kernels retain the
verified computation. Training and fixed-weight interventions keep the
differentiable model calculation.

- The GoPro checkpoints for seeds 42, 43, and 44 and the O-HAZE seed 42 checkpoint
  passed eight small-input comparisons. Model parameters and buffers were equal
  before and after inference. A changed depth parameter was detected.
- The CPU backward pass remained finite. See `supplement/evidence/INFERENCE_PREFLIGHT.json`.
- All 1,111 GoPro test images passed the recorded per-image metric comparison.
  The largest PSNR change was 0.0000048 dB. The mean was 30.5813204 dB.
  See `supplement/evidence/FULL_GOPRO_PARITY.json` and its full per-image file.
- The five O-HAZE images gave 26.7752251 dB. The published context-generator CSPN
  entry gave 23.9342587 dB. Both passed the original per-image metric comparison.
- The separate benchmark entry passed a one-image numerical smoke check and
  48 paired timing calls with four CPU threads. The result is an entry check
  on the current host. The paper retains its three frozen measurement processes.

The original 55-image elementwise panel and the three timing processes remain
in `supplement/benchmark/recorded`. Timing covers one fixed image, not the whole
test set. See `supplement/benchmark/README.md` for the measurement procedure.

## CSPN generator control

The package includes the trained endpoint, initial weights, configuration, and
paired training-data recipe. All 111,600 sample keys matched the frozen run.
Six input/target pairs matched bit for bit after augmentation. The training
configuration matched after changing the data paths and run name. The initial
state loaded with strict keys, and its CPU backward pass was finite.
See `supplement/evidence/CSPN_RECIPE_PARITY.json`.
These checks do not repeat the 55,800-update training run.

## Inventory and accounting

The 2026-09-13 registry had 56 experiment entries, including the existing
no-normalization endpoint with evaluation-only support. Restoration CSPN controls use
the `cspn-propagation` suite. The current-state-center classification and
segmentation comparisons are outside this list. Their archived records remain
outside the release package.

Use `python paper.py verify` for endpoint hashes and `python tools/check_release.py`
for the registered inventory. Use `python -m unittest discover -s tests` and
`python supplement/verify.py` for regression, result, source, and package checks.
The final release check report is stored outside the package to avoid changing
the checked package inventory.

The MAC counter excludes normalization, pooling, and interpolation. The raw
counts and removed terms are in `supplement/evidence/MAC_COUNT_RECONCILIATION.json`.
This accounting change did not change weights or quality metrics.

The portable Spatial-Mamba Linux build branch is not tested. Third-party license
coverage is described in `supplement/THIRD_PARTY.md`.

All 56 entries registered on 2026-09-13 passed hash and strict-load checks. The existing nine
regression tests passed. Spatial-Mamba required its existing local compiled
extensions for import. The official scan was not run on CPU. See
supplement/evidence/default_native_20260913/RELEASE_LOAD.json.

## Direct SSIM release update, 2026-09-19

The release now links current direct RGB SSIM records and excludes historical
pooled-only rows from recorded collection. It adds the completed Haze4K endpoints,
the GoPro joint ablation, and the two O-HAZE full-block transfers.

All 14 Haze4K checkpoints passed strict CPU loading; the 13 models with a CPU
forward passed a finite small-image output check. SS2D loading was checked without
running its CUDA scan on CPU. A separate source review checked the Haze4K model
configurations, data order, loss, optimizer, scheduler, and metric calculation
against the original runs. It caught and restored the full model fused-AdamW
setting before publication. The runtime snapshot matches all 112 source hashes.

The per-step kernel-regeneration endpoint also passed a bitwise source-to-stage
CPU parity check on a 16-by-20 input. All 73 registry entries pass inventory and
checkpoint hash validation; all 69 unique weight files match their recorded hashes.

Current package validation checks endpoint hashes, every linked per-image mean,
image IDs, canonical SSIM identity, Python syntax, and absence of machine paths.
The tests and final manifest report are stored outside the checked folder.
Historical fixed-weight diagnostics remain separate from this release package.

## Method and proof map review, 2026-09-23

`THEORY_AND_CODE.md` maps the current method equations to the release operator.
`python tools/check_nsi_equations.py` compares a small explicit grid update
with the CPU operator. It checks kernel gain, update, and weighted readout.
This numerical check does not replace the analytic proof in the paper appendix.
All 14 package regression tests passed. No model was trained or changed.
No full dataset inference was run for this documentation update.
