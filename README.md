# Beyond Sequential Scanning: Rethinking Visual State Propagation with Neighborhood State Interaction for Image Restoration

**Authors:** Jiajun Xu, Hongyuan Jing, Mengmeng Zhang, Qiyu Rong, Tong Xin, Yi Ren, Zhi Liu, Huihui Bai

Official code, experiment records, and Lean 4 formalization for **Neighborhood State Interaction (NSI)**.

## Overview

Many visual state space models (VSSMs) organize 2D state propagation along 1D scan paths, making local interactions between spatially adjacent locations indirect and scan-order dependent. **NSI** replaces sequential scanning with synchronous 2D neighborhood state-space recurrence over coupled channel pairs:

- **Coupled $2\times 2$ rotation--scaling transitions**: Input-conditioned $3\times 3$ pairwise transformations are generated once and shared across propagation steps under a strict gain bound ($\rho = 0.95$), ensuring contractivity.
- **State-dependent retention & incremental readout**: Groupwise radial retention ($g \le 4$ pairs) attenuates accumulated state as magnitude grows, while learned incremental weights combine updates across propagation steps.
- **Machine-checked stability**: Contraction and step-independent state/readout bounds are formally verified in **Lean 4** (`formalization/`).

## Repository Layout

| Path | Description |
|---|---|
| `token_mixers/ours/` | Core NSI token mixer implementation |
| `tasks/restoration/` | VIVNet-based restoration runners (GoPro, O-HAZE, Haze4K) |
| `formalization/` | Lean 4 + mathlib proofs (19 proof files, 203 theorems) |
| `supplement/` | Extended controls, ablation evidence, and verification scripts |
| `diagnostics/` | ERF, module MACs, contraction, and state-trajectory records |
| `configs/`, `splits/` | Training configurations and dataset splits |
| `tools/`, `tests/` | Formula verification (`check_nsi_equations.py`) and unit tests |
| `paper.py`, `reproduce.py` | Suite-level evaluation, training, and metric collection CLIs |

> **Third-party baselines:** To respect upstream licensing, vendored third-party scan baselines (`VMamba`, `MaIR`, `Spatial-Mamba`) are **not** included in this repository. See [supplement/THIRD_PARTY.md](supplement/THIRD_PARTY.md) for pinned upstream URLs and target paths if you wish to run scan-baseline comparisons.

## Quick Start

Verify the NSI formula-to-code equivalence, unit tests, and experiment registry locally (no GPU or dataset required):

```bash
python tools/check_nsi_equations.py
python -m unittest discover -s tests
python paper.py list --suite current-restoration
```

See [THEORY_AND_CODE.md](THEORY_AND_CODE.md) for the exact correspondence between paper equations, PyTorch implementation, and Lean 4 declarations.

## Lean 4 Formalization

All stability theorems (retention derivative, $g \le 4$ non-expansiveness, global contractivity, state/readout norm bounds, and $g \ge 5$ expansion counterexamples) are formalized under Lean `v4.32.2` in [formalization/](formalization/README.md). To rebuild and verify all axiom dependencies locally:

```bash
cd formalization
lake exe cache get
python check_all.py
```

## Pretrained Weights & Full Reproduction

Pretrained checkpoints (63 tensor files + SHA-256 checksums) are hosted on [Google Drive](https://drive.google.com/drive/folders/1qr1F9wgUWarD5YrJRK1ZU-_zyHSVCTIR). Download and extract the weights alongside this repository, then follow [REPRODUCE.md](REPRODUCE.md) for environment setup, dataset preparation, and full evaluation/training commands.

## Citation

```bibtex
@article{xu2026nsi,
  title   = {Beyond Sequential Scanning: Rethinking Visual State Propagation with Neighborhood State Interaction for Image Restoration},
  author  = {Xu, Jiajun and Jing, Hongyuan and Zhang, Mengmeng and Rong, Qiyu and Xin, Tong and Ren, Yi and Liu, Zhi and Bai, Huihui},
  journal = {arXiv preprint},
  year    = {2026}
}
```
