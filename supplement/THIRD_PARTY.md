# Third-Party Sources

To respect upstream licensing, third-party scan baseline implementations (`VMamba`, `MaIR`, `Spatial-Mamba`) are **not redistributed** in this repository. All third-party files remain under their original authors' licenses and copyrights.

## Baseline Repositories (Required Only for Third-Party Scan Comparisons)

If you want to run or build the third-party scan baselines (`SS2D`, `MaIR`, `Spatial-Mamba`), fetch the upstream repositories at the pinned commits below and place them into the expected local paths:

| Method / Component | Upstream Repository | Pinned Commit | Target Local Path(s) |
|---|---|---|---|
| **VMamba** (`SS2D` / `selective_scan`) | https://github.com/MzeroMiko/VMamba | `2ed52ead062a51a64521ed3871d52914bf532876` | `tasks/restoration/runtime/third_party/VMamba`<br>`tasks/restoration/haze4k/runtime/third_party/VMamba` |
| **MaIR** (`VMM` / `NSS` / `SSA`) | https://github.com/XLearning-SCU/2025-CVPR-MaIR | `07d58af10175f52bba0bb6f82edba1ea5d891e12` | `tasks/restoration/runtime/third_party/2025-CVPR-MaIR`<br>`tasks/restoration/haze4k/runtime/third_party/2025-CVPR-MaIR` |
| **Spatial-Mamba** (`StructureAwareSSM`) | https://github.com/EdwardChasel/Spatial-Mamba | `f72ed9b2486a5931190912dcfa8b964033c80e8c` | `supplement/third_party/SpatialMamba` |

## Backbone and Framework Attributions

- **BasicSR**: The restoration training/evaluation harness in `tasks/restoration/` builds on [BasicSR](https://github.com/XPixelGroup/BasicSR) (Apache-2.0 License).
- **VIVNet**: Controlled token-mixer comparisons share the U-shaped restoration backbone from [VIVNet](https://github.com/c-yn/VIVNet) (Cui et al., *IEEE TPAMI*, 2026).
- **EVSSM**: Native Mamba backbone experiments build on [EVSSM](https://github.com/kkkls/EVSSM) (Kong et al., *CVPR*, 2025).
- **BPNet**: Upstream source reference noted in prior audits: https://github.com/ginobilinie/BPNet (not bundled in this repository).
