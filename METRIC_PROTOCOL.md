# Restoration SSIM protocol

All current restoration results use metric protocol `direct_rgb_ssim_v1`.
The metric compares the RGB prediction and target directly. It does not pool
either image before SSIM. The canonical result field is `ssim`.

The common SSIM settings are an 11 by 11 Gaussian window, sigma 1.5, K1 0.01,
and K2 0.03. The calculation uses valid-window statistics. Each dataset keeps
its established image scale and numerical backend:

| Dataset | Images scored | Backend | Data range | Metric precision |
|---|---|---|---:|---|
| O-HAZE | IDs 41 to 45, resized to long side 512 | `pytorch_msssim.ssim` | 1 | FP32 |
| GoPro | 1,111 native-size RGB frames | direct PyTorch valid convolution | 1 | FP32 |
| Haze4K | 1,000 native-size RGB images | NumPy/OpenCV valid Gaussian window | 255 | float64 |

For O-HAZE, `direct` means that no image pre-pooling occurs. It does not mean
original-resolution scoring. O-HAZE keeps the paper's long-side-512 rule.

New result files contain a `metric_protocol` object. Its `id` must be
`direct_rgb_ssim_v1`, and `image_pre_pooling` must be `false`. The backend,
data range, and precision fields record the numerical implementation. They do
not change the shared metric meaning.

Some bundled GoPro records were created before the `metric_protocol` field was
added. The collector accepts these records only when their per-image hash is
present and their recorded inference protocol is native. New GoPro results
write the full protocol object.

The O-HAZE reassessment is in
`supplement/evidence/ssim_reassessment_20260912`. It contains 38 groups and
190 image records. The experiment registry uses 21 reassessed conditions.
Five historical conditions have no direct SSIM measurement:

- `ohaze-step_conditioned_rules-seed42`
- `ohaze-equal_parameter_real-seed42`
- `ohaze-alternative_generator-seed42`
- `ohaze-conjugate_symmetric-seed42`
- `ohaze-p0t_diagonal_delay-seed42`

Their old values remain only in `legacy_pooled_ssim`. Recorded collection lists
these rows as excluded and does not substitute the historical values.

The O-HAZE reassessment files also contain CPU float64 replay PSNR. The paper
keeps its formal FP32 PSNR. During recorded collection, the collector reads the
published PSNR from `legacy_published.psnr` and reads direct SSIM from `ssim`.
It never reads `legacy_published.ssim` as a current result.

The CPU test in `tests/test_direct_ssim.py` compares the two PyTorch SSIM
backends on the same unpooled tensor with a tolerance of 2e-6. It also checks
the 38-group, 190-image reassessment inventory and all 23 registered O-HAZE
direct SSIM values.
