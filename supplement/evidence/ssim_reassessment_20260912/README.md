# Direct SSIM reassessment

This record uses O-HAZE test images 41 to 45. Each image has equal weight.
The SSIM calculation uses an 11 by 11 Gaussian window with sigma 1.5.
It uses K values 0.01 and 0.03 and data range 1. It does not pool the images before SSIM.
See `protocol.json` for the full metric and scale rules.

`groups` contains 38 result files. Each `FINAL.json` contains all five image scores,
the measured PSNR, direct SSIM, replayed pooled SSIM, and checkpoint SHA256.
`all_affected_summary.json` and `all_affected_per_image.csv` collect these 190 records.
Historical published values remain in `legacy_published`.
The published long-side-512 PSNR values are unchanged in the experiment registry.
The measured PSNR in this evidence uses CPU float64 MSE.
The formal evaluator retains its original FP32 PSNR calculation.

`native_scale_per_image.json` and `native_scale_per_image.csv` contain 160 scale cells.
`native_scale_summary.json` contains their means and paired changes.
Input, output, and target use the same size in each scale cell.
The output is not mapped to long side 512 before scoring.

The registry has 21 updated direct SSIM values. Five historical conditions have no
new direct SSIM measurement. Their former values are in `legacy_pooled_ssim` only.
GoPro values are unchanged.

Source record IDs and hashes refer to the archived evaluation records.
Those raw records, prediction arrays, model binaries, and dataset images are not
included here. This export contains no private machine paths.
No model forward operation or training update was run to create this export.
