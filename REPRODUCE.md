# Reproduce the paper

This package contains the code, records, and weights for the paper experiments.
The release has a code-and-records ZIP and separate weights. The cloud copy
of the weights uses smaller ordinary ZIP files because of the upload size limit.
Extract the code ZIP and all weight ZIP files into the same parent folder.
All archives use the code/ root. The local single weights ZIP contains the same
63 tensor files and can be used instead of the cloud files.
Use Python 3.12 and an NVIDIA GPU for training and evaluation.
Use `paper.py` to run a complete table with one command.
The default action in `reproduce.py` is evaluation.

## Download the weights

Download all 16 weight ZIP files from [Google Drive](https://drive.google.com/drive/folders/1qr1F9wgUWarD5YrJRK1ZU-_zyHSVCTIR).
The folder includes `WEIGHTS_README.txt` and `WEIGHT_SHA256SUMS.txt`.
Extract all weight ZIP files into the same parent folder as the code ZIP.
These are ordinary ZIP files. Do not join them as binary volumes.

## Start with the current restoration paper

Use `python paper.py list --suite current-restoration` to see active restoration entries.
Use `python paper.py verify --suite current-restoration` to check their weights.
Use the commands in Section 4 to evaluate models.
Use [THEORY_AND_CODE.md](THEORY_AND_CODE.md) to match the equations to the code and proof.
This release contains image restoration only. Use the current-restoration
suite for the 55 current paper entries. Historical restoration controls and
duplicate native-evaluation aliases remain available for audit.

## Lean proofs

See [formalization/README.md](formalization/README.md) for the proof sources,
pinned Lean and mathlib versions, build commands, and the paper-to-theorem map.
Run `python check_all.py` from `formalization/` after installing its dependencies.
The check rebuilds every local proof and checks all local declarations for
unproved placeholders and extra axiom dependencies. The release verification
records are in `formalization/verification/`.


## 1. Install the environment

Open a terminal in this directory.
Use a separate environment for this package.
The commands below use the PyTorch 2.10 CUDA 13.0 build.
Your NVIDIA driver must support this CUDA version.

```text
python -m venv .venv
```

Activate the environment on Linux:

```sh
. .venv/bin/activate
```

Activate the environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages:

```text
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The default FP32 inference path uses Triton 3.6. The requirements file selects
the Windows or Linux package. The checked Windows build is 3.6.0.post26.

SS2D and MaIR use the official selective-scan CUDA extension.
Fetch the upstream third-party repositories per [supplement/THIRD_PARTY.md](supplement/THIRD_PARTY.md) before building.
Install the CUDA toolkit before you build this extension.
On Linux, run:

```text
python -m pip install --no-build-isolation ./tasks/restoration/runtime/third_party/VMamba/kernels/selective_scan
python -c "import selective_scan_cuda_oflex; print('PASS')"
```

Windows requires a matching C++ compiler, CUDA toolkit, and Triton installation.
Use Linux if the extension does not build on Windows.
Do not replace selective scan with a different operator.

## 2. Check the package

Extract both ZIP files before these checks. These commands do not train a model.
The code ZIP alone can list entries and collect the included result records.
The first command lists the paper experiments.
The second command checks each weight file against its SHA-256 value.

```text
python paper.py list --suite current-restoration
python paper.py verify --suite current-restoration
python tools/check_release.py --load
```

The last command constructs each model and checks its weight keys and tensor shapes.
See `experiments.json` for the exact experiment list.
See `PAPER_RESULTS.md` for the paper metrics and selection rules.
See `RELEASE_CHECKS.md` for the checks completed on this package.
See [supplement/README.md](supplement/README.md) for the supplementary component,
native-image, Spatial-Mamba, and FP32 timing experiments.
The main GoPro results use one native full-image forward with no extra outer padding.
Historical tiled results remain in their recorded folders with separate protocol labels.
The full radial restoration model selects joint FP32 inference in `eval()` under
`torch.inference_mode()` or `torch.no_grad()` on CUDA. A separate preparation call
is not required. CPU, training, gradients, and autocast use the model calculation.
The depth cache refreshes after weights change. The GoPro evaluator and benchmark
default to native images; use `--inference-mode tiled` for the historical tiling protocol.
See [supplement/benchmark/README.md](supplement/benchmark/README.md) for the checks and timing scope.


## 3. Prepare the data

Download each dataset from its official source.
Dataset images are not included in this package.
Put the extracted source folders in `source_data`.
Use the names in these commands.

```text
python reproduce.py --task restoration --dataset ohaze --action prepare --source-root source_data/ohaze --data-root data/ohaze
python reproduce.py --task restoration --dataset gopro --action prepare --source-root source_data/gopro --data-root data/gopro
```

Add `--copy-data` if the source and output are on different drives.
The default operation creates hard links.
Keep the source files unchanged after you create hard links.

| Dataset | Required source layout | Paper split |
|---|---|---|
| O-HAZE | `GT` and `hazy`, with the official image names | IDs 1 to 40: train; IDs 41 to 45: test |
| GoPro | `train/<sequence>/{blur,sharp}` and `test/<sequence>/{blur,sharp}` | 2,103 train pairs; 1,111 test pairs |
| Haze4K | `train/{haze,gt}` and `test/{haze,gt}` under `data/haze4k` | 3,000 train pairs; 1,000 test pairs |


## 4. Reproduce a table

Each command selects all required weights and seeds.
Run each command once in a new output directory.

```text
python paper.py eval --suite ohaze-main
python paper.py eval --suite ohaze-ablations
python paper.py eval --suite gopro
python paper.py eval --suite haze4k
```

Use this command to evaluate all current paper entries:

```text
python paper.py eval --suite current-restoration
```

Add `--data-root /path/to/data` to use another data directory.
That directory must contain `ohaze`, `gopro`, and `haze4k` as required by the selected suite.
Add `--output /path/to/results` to use another output directory.
Add `--dry-run` to check the commands without model execution.

Collect a table after evaluation:

```text
python paper.py collect --suite ohaze-main
python paper.py collect --suite gopro
```

The command writes `runs/paper/SUMMARY.csv` and `runs/paper/SUMMARY.json`.
It reports missing results as an error.
GoPro statistics use the mean and sample standard deviation of three seeds.
Collection replaces these summary files for the selected suite.
Use `python paper.py collect` after all suites finish to collect all results.

## 5. Evaluate one model

```text
python reproduce.py --task restoration --dataset ohaze --mixer ours --action eval --data-root data/ohaze
python reproduce.py --task restoration --dataset ohaze --mixer mair --action eval --data-root data/ohaze
python reproduce.py --task restoration --dataset ohaze --mixer ours --ablation p0t_diagonal_delay --action eval --data-root data/ohaze
python reproduce.py --task restoration --dataset ohaze --mixer ours --ablation p0s_state_dynamic --action eval --data-root data/ohaze
```

Use `--checkpoint path/to/file` to select a checkpoint explicitly.
Use `--run-dir path/to/results` to select a separate result directory.

## 6. Train from the initial state

Replace `eval` with `train` to train a complete suite.
Use a new output directory.

```text
python paper.py train --suite ohaze-main --output runs/new-training
python paper.py train --suite ohaze-ablations --output runs/new-training
python paper.py train --suite gopro --output runs/new-training
```

Evaluate the new weights from a completed training suite:

```text
python paper.py eval --suite ohaze-main --trained --output runs/new-training
python paper.py collect --suite ohaze-main --output runs/new-training
```

The `--trained` option selects the weights in the training output directory.
Without this option, `paper.py eval` selects the released paper weights.

To train one model and then evaluate its new weights, run:

```text
python reproduce.py --task restoration --dataset ohaze --mixer ours --action all --data-root data/ohaze --run-dir runs/new-ours
```

Training does not load the released weights.
A new run can differ because of hardware, library versions, and random operations.
P0-T resumed its own delayed run at update 45,000 in the original experiment.
The original run did not save the data-loader state or random-generator state.
A new P0-T run does not reproduce the original batch sequence exactly.

## 7. Run fixed-weight diagnostics

See `DIAGNOSTICS.md` for the commands and intervention definitions.
These tests use existing weights and do not train a model.
Their results are separate from the independent-training ablations.

## 8. Count parameters and MACs

```text
python reproduce.py --task restoration --dataset ohaze --mixer ours --action profile
python reproduce.py --task restoration --dataset ohaze --mixer mair --action profile
python reproduce.py --task restoration --dataset ohaze --mixer ours --ablation p0s_state_dynamic --action profile
```

The command writes a full-model result in `profiles`.
One multiply-accumulate is one MAC.
The counter includes convolutions, linear layers, matrix products, selective scan, and local propagation.
It excludes normalization, activations, gates, padding, and rearrangement.
The propagation term `O(9KDN)` is not the full-model cost.
`K` is the number of synchronous state updates.

## Package contents

| Path | Function |
|---|---|
| `reproduce.py` | Run one experiment |
| `paper.py` | Run or collect a paper table |
| `experiments.json` | Map paper settings to commands and weights |
| `configs` | Store the training settings |
| `tasks` | Store the carrier models and task code |
| `token_mixers` | Store the shared token mixers |
| `weights` | Store the selected paper weights and SHA-256 values |
| `splits` | Store the fixed data lists |
| `tools` | Check the package and run diagnostics |

MaIR means **MaIR-VMM (matched slot)** throughout this package.
It uses the official VMM in the common VIVNet carrier.
It is not the full MaIRUNet model.

The documentation uses short instructions and consistent technical terms from ASD-STE100.
Software names, model names, tensor notation, and command options are technical terms.
Upstream copyright notices keep their original wording.
See [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf) for the language standard.

The `cspn-propagation` suite contains the restoration CSPN controls. It includes
the O-HAZE context-generator condition, its fixed initial weights, and its endpoint.
The mismatched current-state-center CSPN results are outside the active paper list.

## Current restoration results and metric identity

The current restoration manuscript uses direct RGB SSIM: an 11-by-11 Gaussian
window with sigma 1.5, valid windows, K1=0.01 and K2=0.03. No image pooling is
applied before SSIM. O-HAZE uses the stated long-side-512 evaluation images;
GoPro and Haze4K use native whole images. The metric does not resize them again.
Each new evaluation writes `metric_protocol.id = direct_rgb_ssim_v1` and a
canonical `ssim` field. See `METRIC_PROTOCOL.md` for the numerical backends.

Historical pooled SSIM records remain identifiable in the evidence archive.
The active collector resolves the direct reassessments and excludes the five
historical entries without a direct measurement. Exclusion is reported in
`SUMMARY.json`; a missing current result remains an error.

Haze4K now includes all 14 completed endpoints used by the manuscript. The GoPro
joint ablation is registered as `gopro-joint-seed42`. See
[Haze4K instructions](tasks/restoration/haze4k/README.md) for the data manifest,
training protocol, and evaluation commands.

To collect the current restoration endpoints without running inference:

```text
python paper.py collect --suite current-restoration --recorded --output runs/current-records
```

This suite excludes duplicate native-evaluation aliases and historical conditions
without direct SSIM. The underlying raw evidence remains available.

The ohaze-full-blocks suite provides fixed-checkpoint evaluation for the MaIR
and Spatial-Mamba complete blocks. Their original 55,800-step training settings
are included in the configs. These two release entries are evaluation-only;
the common training launcher reports them as skipped.

See [diagnostics/README.md](diagnostics/README.md) for the current ERF, module-cost, state-trajectory, and contraction records and scripts.

See [RELEASE_AUDIT.md](RELEASE_AUDIT.md) for the September 25 release audit.
