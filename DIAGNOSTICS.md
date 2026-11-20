# O-HAZE fixed-weight diagnostics

This file describes the CPU diagnostics for the O-HAZE checkpoint.
The diagnostics do not train a model. They do not change a checkpoint.
They do not use CUDA.

The SSIM values below come from historical CPU probes.
Do not compare them with the current direct RGB SSIM table.
Use `METRIC_PROTOCOL.md` and `PAPER_RESULTS.md` for the current metric.

## Run the paper settings

Run from the reproduction code directory. Pass the data and checkpoint paths
that you want to inspect.

```powershell
python tools/diagnose_ohaze.py `
  --mode all `
  --release-root . `
  --data-root data\ohaze `
  --checkpoint weights\ohaze\ours\seed42\last.pth `
  --output-dir diagnostic_outputs\ohaze_fixed_weight
```

The command writes `run.json`, `delay.json`, `delay.md`,
`normalization.json`, and `normalization.md` to the output directory.
Use `--mode delay` or `--mode normalization` to run one probe. Use
`--limit 1` for a short smoke run. A one-image run is not the paper result.

The script uses the release configuration by default:

```text
configs/restoration/ohaze/O-HAZE_VIVNet_RadialProfiled_K5_P256.yml
```

Use `--config` when the checkpoint uses another compatible configuration.
The script keeps all tensors on the CPU, even when a CUDA device is present.

## Delay probe

The delay probe follows:

```text
research_outputs/diagonal_recurrent_timing_cpu_20260904/run_probe.py
```

It evaluates three fixed-weight settings on the same five O-HAZE test pairs:

| Setting | Definition |
|---|---|
| `baseline` | All nine taps read the current pre-transition state. |
| `diagonal_recurrent_lag1` | The four diagonal taps read the previous state from step 2 onward. |
| `axial_recurrent_lag1` | The four axial taps read the previous state from step 2 onward. |

The first transition is unchanged in both lag settings. The model keeps the
learned coefficients, row-mass projection, K=5 depth readout, tap count, and
arithmetic count.

The existing five-image CPU result is:

| Setting | PSNR | SSIM | PSNR change from baseline |
|---|---:|---:|---:|
| `baseline` | 26.7752 | 0.94682 | +0.0000 dB |
| `diagonal_recurrent_lag1` | 25.6392 | 0.93304 | -1.1360 dB |
| `axial_recurrent_lag1` | 26.0175 | 0.94135 | -0.7577 dB |

This is a fixed-weight timing diagnostic. It tests the use of timely
recurrent two-dimensional neighbors. It is not SS2D emulation. It is not a
matched retraining comparison.

## Coefficient normalization probe

The normalization probe follows:

```text
research_outputs/row_mass_alternatives_cpu_20260904/run_probe.py
```

It keeps the trained checkpoint and replaces only the coefficient projection
at inference time. The four settings are:

| Setting | Definition |
|---|---|
| `row_mass` | The trained per-row complex L1 projection with `rho_max`. |
| `global_mean_mass` | One positive scale per operator and input. It matches the projected mean row mass. |
| `kernel_rmsnorm` | RMS normalization of the nine complex taps per row, followed by mean-mass scaling. |
| `kernel_layernorm` | LayerNorm of the 18 real coordinates per row, followed by mean-mass scaling. |

The existing five-image CPU result is:

| Setting | PSNR | SSIM |
|---|---:|---:|
| `row_mass` | 26.7752 | 0.94682 |
| `global_mean_mass` | 8.3424 | 0.52648 |
| `kernel_rmsnorm` | 25.7580 | 0.93947 |
| `kernel_layernorm` | 3.9665 | 0.25799 |

The script also reports a centered 7x7 operator probe for all 22 operators.
It measures row mass, rows above `rho_max`, and exact induced infinity norms
for depths 1, 3, 5, and 7. These matrix results are a fixed-weight probe.
They do not show that a new normalization would fail after retraining.
The paper does not report independent Kernel RMSNorm retraining.

## Metric and data contract

The script uses the release functions in `tasks/restoration/evaluate_ohaze.py`.
For each pair it:

1. converts the RGB image to `[0, 1]` FP32;
2. resizes the long side to 512 pixels with bicubic interpolation;
3. pads the image to a multiple of 32 with reflect padding;
4. evaluates the full image; and
5. crops the output to the unpadded size before PSNR and SSIM.

The formal paper setting uses five disjoint O-HAZE test pairs. The data root
must contain `test/hazy` and `test/gt` with matching file names.

## Other evidence and current scope

The following reports are real sources in the research workspace. The
wrappers below use the same source logic and write new results only when you
run them:

* `research_outputs/diagonal_path_lag_cpu_20260904` tests lag 1, 2, and 5,
  axial lag, and an extra zero-weight warm-up update. It is a separate
  fixed-weight path diagnostic.
* `research_outputs/diagonal_restoration_information_cpu_20260904` is a
  closed-form ridge probe of local RGB residual information. It is not a
  network endpoint comparison.
* `research_outputs/direct_path_performance_cpu_20260904/run_probe.py` is the
  research source for the bundled `tools/diagnose_synthetic.py` logic.
* `research_outputs/trajectory_readout_temporal_alignment_cpu_20260904/run_probe.py`
  is the research source for the bundled `tools/diagnose_trajectory.py` logic.
* `research_outputs/context_halo_cpu_20260904/run_probe.py` and
  `run_grid_probe.py` are the research sources for the bundled
  `tools/diagnose_halo.py` logic.

All three wrappers are CPU only. They load a saved model or saved outputs.
They do not train a model or change a checkpoint.

## Controlled synthetic path probe

Run the deterministic mechanism check without a model:

```powershell
python tools/diagnose_synthetic.py `
  --mode synthetic `
  --output diagnostic_outputs\synthetic_path.json
```

The default matches the source protocol: seed `20260904`, 192 train fields,
64 test fields, a 16x16 grid, and blur levels 0, 1, and 3. The reported
source result is **SUPPORTED**. Direct K1 reaches the diagonal target. The
delayed K1 control has NMSE 1.0012, 0.2973, and 0.07404 at blur levels 0, 1,
and 3. Delayed K2 returns to the direct result. The axis negative control is
unchanged.

The saved-output association can be run with explicit paths:

```powershell
python tools/diagnose_synthetic.py `
  --mode orientation `
  --release-root . `
  --suite-root runs\paper `
  --data-root data\ohaze `
  --output diagnostic_outputs\synthetic_orientation.json
```

The `runs\paper` folder must contain the saved O-HAZE endpoint images. Create
these images with the release evaluator before this association check:

```powershell
python paper.py eval --suite ohaze-main --data-root data --output runs\paper
```

This image-generation step uses the normal release evaluator. The association
check itself stays on CPU and does not train a model.

The existing five-image source result has mean diagonal-minus-axial contrast
`+1.4624%`, but only 3/5 positive images. Its verdict is **NOT_SUPPORTED**.
This is a fixed-output association check. It is not a matched retraining
comparison.

## Trajectory readout probe

Run the bundled trajectory and readout permutation test. The wrapper contains
the fixed-weight probe logic and uses only release-relative paths:

```powershell
python tools/diagnose_trajectory.py `
  --release-root . `
  --data-root data\ohaze `
  --checkpoint weights\ohaze\ours\seed42\last.pth `
  --output-dir diagnostic_outputs\trajectory_readout
```

Use `--limit 1 --threads 1` for a CPU smoke run. The source five-image
result is **SUPPORTED**: baseline PSNR is 26.7752 dB, reverse readout is
13.5486 dB, cyclic readout is 24.1319 dB, and uniform readout is 21.7869 dB.
The two permutation mean change is -7.9349 dB on 5/5 images. The negative
control has zero output error. The source records mean learned depth weights
0.2709, 0.3923, 0.6355, 1.1737, and 2.5276.

This is a fixed-weight causal diagnostic. It does not prove a retrained
architecture effect or a super-additive interaction with retention.

## Context halo consistency probes

Run the centered-core test:

```powershell
python tools/diagnose_halo.py `
  --mode center `
  --release-root . `
  --data-root data\ohaze `
  --checkpoint weights\ohaze\ours\seed42\last.pth `
  --output-dir diagnostic_outputs\context_halo
```

Use `--max-images 1 --threads 1` for a CPU smoke run. The source five-image
center result is **NOT_SUPPORTED**: full minus halo64 is `+0.3486` dB but is
positive on only 2/5 images. The mean core PSNR values for halos 32, 64, 96,
128, and full are 28.0598, 28.5120, 29.2378, 28.8465, and 28.8606 dB.

Run the post-result spatial average over all aligned cores with:

```powershell
python tools/diagnose_halo.py `
  --mode grid `
  --release-root . `
  --data-root data\ohaze `
  --checkpoint weights\ohaze\ours\seed42\last.pth `
  --output-dir diagnostic_outputs\context_halo
```

The source grid result evaluates 480 cores. Full minus halo64 is `+0.5180`
dB, with positive image means on 5/5 images, so its verdict is **SUPPORTED**.
This result supports useful context outside a 64-pixel halo across the image.
It does not erase the centered result or assign the whole gain to one model
component.
