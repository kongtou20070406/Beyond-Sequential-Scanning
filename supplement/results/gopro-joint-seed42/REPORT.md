# GoPro joint ablation endpoint

This record publishes the completed seed-42 joint arm (identity retention plus
final-state/uniform readout) from campaign `gopro_joint_seed42_20260917`.

## Provenance

- Source config: `work/configs/joint.yml`, SHA-256 `0550016d4f387313a7e87849876ec7e7b6f26c6983c978d05f4f306222542cdf`.
- Published portable config: `configs/supplement/gopro-joint-seed42.yml`, SHA-256 `93ab80aadd530e37b1763367bdddc2e646871349f732b769b0c5ef4f45ba22ea`; `__DATA_ROOT__` is the only dataset root placeholder.
- Source runtime manifest: `SOURCE_MANIFEST.json`, SHA-256 `63069ca663ceab9b701b7f9843076cc880d28b8050fa7b35b8d4d09805297334`; the published `supplement/runtime_manifests/joint.json` retains this byte hash.
- Endpoint source and published weight SHA-256: `00981e007389a293da45d5f0159da0cd5840ee607e841a19f8c41f90a257ad94` (23,409,533 bytes).
- Shared initialization reference SHA-256: `d503cb0eadd00a9588df3d22a99b95d0d807d6a18e01798ba8d64c06b1762fc8`.
- Source evaluation `FINAL.json` SHA-256: `8abe610d7a5ab03b968c75f15319ce9a9d8ea8130e949e093a6217753fcf3836`.
- Published per-image SHA-256: `e11f357e697752a969f78c1495e630c89de915bb448435dcad32f5a64c762c45`; source `IDENTITY.json` SHA-256 `208aefa96d802993269d28342148840e15f1a5ddc4adee1a2fafe9ac9375251f`; source `per_video.json` SHA-256 `15f83ec2cfeb8e50535088c49966763ebbc044408201dffba28ec4c78b794855`.

The joint overlay contains the seven source files that differ from the current
base runtime: `VIVNet_ProfiledCLPO_arch.py`,
`VIVNet_RadialProfiledAblation_arch.py`, `base_model.py`,
`image_restoration_model.py`, `clpo2027/model.py`,
`clpo2027/selective_f9_backend.py`, and `clpo2027/timing_backend.py`.

## Result

The fixed endpoint evaluates all 1,111 original GoPro images at native full
resolution with FP32 and TF32 disabled:

- PSNR: `30.87720270525969` dB
- direct RGB SSIM: `0.9125754498448273`
- per-image count: `1111`
- protocol: `direct_rgb_ssim_v1`, RGB, no pre-pooling, Gaussian window 11,
  sigma 1.5, `data_range=1`, `k1=0.01`, `k2=0.03`, backend
  `torch.conv2d.valid`, numeric precision `FP32`

## Checks

The CPU-only packaging check parsed all JSON/JSONL files, found 1,111 rows in
11 video groups, matched every group count against `per_video.json`, reproduced
the per-image SHA above, and reproduced the endpoint and initialization hashes.
No GPU execution or training was started during packaging.
