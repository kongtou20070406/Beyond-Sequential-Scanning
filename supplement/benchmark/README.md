# FP32 inference

The paper's restoration evaluation commands use joint input heads and direct
five-step propagation for the full radial model. This is the default CUDA FP32
path. It uses the released weights and keeps TF32 off. It does not change training.

The kernels are in `tasks/restoration/runtime/clpo2027/fp32_heads.py` and
`fp32_rollout.py`. Training and fixed-weight intervention tools use the
differentiable model computation. The model selects this path automatically during CUDA FP32 inference.
The depth cache refreshes after weights change. Use
`prepare_fp32_inference(model, enabled=False)` for a reference comparison.

```text
python supplement/benchmark/run_fusion.py verify --data-root data/gopro --output runs/fp32-panel
python supplement/benchmark/run_fusion.py benchmark --data-root data/gopro --output runs/fp32-timing
python supplement/benchmark/run_fusion.py evaluate --data-root data/gopro --output runs/fp32-full
```

Use a new output directory for each command. Add `--dry-run` to check the selected
weights without running inference. Add `--limit 1` to `verify` for a small smoke
check. The full numerical panel contains 55 images. The evaluation command uses
all 1,111 GoPro test frames. No command starts training.

The benchmark warms up both models and runs 12 rounds for a 512-square crop and
for the fixed native image (720 by 1280). It alternates the two methods. Use
`--start-with ss2d` to reverse the first order in a new process. Time excludes
decode, host transfer, and metrics. The native measurement is one model call. Add `--inference-mode tiled` to use
the historical tiled protocol, including tile extraction and assembly.
The benchmark is a fixed-image comparison, not a test-set mean.

`recorded` contains the three original measurement processes and the numerical
checks for the selected implementation. Each original process also measured an
intermediate implementation in its balanced order. Those original records keep
their source identities. `CURRENT_SUMMARY.json` reports only the selected
implementation and SS2D. No process was selected by its timing result.

## Current native validation

The 2026-09-13 native panel passed all 55 images against the reference forward.
Maximum differences were 1.2516976e-6 per output element, 2.3841858e-6 dB PSNR,
and 1.1920929e-7 SSIM. This is a numerical panel, not a new full-test evaluation.
The recorded main quality values still refer to the 1,111-image reference runs.

Three independent processes each ran 12 alternating method pairs after warmup.
Native CUDA medians were 0.4935363 to 0.4961464 s for ours and 0.5126948 to
0.5144204 s for SS2D. Same-process latency reductions were 3.41% to 3.74%.
Peak allocation increments were 7604.3169 MiB for both methods. These are fixed-image
measurements on the shared RTX 5090 host, not dataset-wide latency averages.
See `../evidence/default_native_20260913/SUMMARY.json` and its raw records.
The files in `recorded` and `CURRENT_SUMMARY.json` retain the historical tiled tests.
