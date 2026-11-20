# Submission package review: 2026-09-25

The submission copy contains restoration code, records, and separate weights.
The training code, model code, configurations, and checkpoint bytes match the
verified source release. This review did not change the NSI calculation.

## Scope and repairs

- Include all 55 current restoration conditions.
- Retain five historical O-HAZE controls and six native-evaluation aliases for
  audit. The current-restoration suite excludes these extra registry entries.
- Include 63 unique tensor files: 55 current endpoints, five historical control
  endpoints, CSPN and GoPro initialization files, and one MaIR checkpoint comparison.
  The GoPro initialization is in supplement/reference and is needed for training.
- Exclude the historical classification and segmentation tasks and weights.
- Correct the stale weight count and release instructions.
- Include the current ERF, module MAC, state-trajectory, and contraction
  diagnostics. Convert local paths to portable paths. Preserve measured values,
  gradient arrays, and original checkpoint hashes.

## Checks completed

- SHA-256 and size checks pass for every included checkpoint.
- Strict model loading passes for all 66 registry entries and all three extra files.
- 56 small CPU forwards pass. Thirteen load cases need CUDA extensions for
  forward execution and were checked by strict loading only.
- All 14 regression tests pass.
- The explicit NSI recurrence and the implementation differ by at most
  1.12e-8 in the seeded CPU check.
- A separate float64 retention-Jacobian check differs by at most 1.39e-17.
- All 55 current result records collect without missing entries. The direct
  RGB SSIM contract is checked. Historical pooled SSIM is kept separate.
- Twenty saved ERF maps match their checkpoint identities and r90 values.
- The portable CPU MAC scripts reproduce the saved module counts and the
  25.1 percent and 30.2 percent rounded complete-model reductions.
- Text and checkpoint metadata scans found no local user paths. The scan for
  submission-author names found no matches. Upstream attribution is retained.

See audit/ for the machine-readable checks and diagnostics/ for added evidence.
Run supplement/verify.py after extracting code and weights to verify the static
file inventory and result records. Use a fresh extraction for this check, before
commands that create output files. The code ZIP alone can run paper.py list and
paper.py collect --suite current-restoration --recorded without the weights.

## Lean proof check added on 2026-09-25

- Include 18 mathematical Lean source files and one declaration-audit file.
- Rebuild all local proofs with Lean 4.32.2 and mathlib commit
  905b95818eb32af7874a58b427f50c1711a5e96c.
- All 19 files pass with warnings treated as errors.
- Check 254 local declarations, including 203 theorems. Their transitive axiom
  dependencies contain only propext, Classical.choice, and Quot.sound.
- Include portable build instructions, source hashes, compiler logs, and the
  exact correspondence to the paper in formalization/.

## Scope of the checks

This review did not rerun full-dataset CUDA inference or training. The ERF
records use one test image and one output pixel per dataset, not dataset means.
The numerical derivative checks are separate from the Lean proof check.
The Lean coverage and its assumptions are stated in formalization/README.md.
MaIR and BPNet license coverage remains unresolved in the existing upstream
source record; see supplement/THIRD_PARTY.md. No new license is assigned here.
