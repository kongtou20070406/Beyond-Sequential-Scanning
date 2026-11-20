# Haze4K reproduction

This folder contains the 14 completed Haze4K configurations used by the current
restoration manuscript. Each model uses seed 42 and 100,000 updates. Parameters
and test results refer to that final endpoint.

Place the original Haze4K folders under `data/haze4k`:

```text
data/haze4k/train/haze/*.png
data/haze4k/train/gt/*.png
data/haze4k/test/haze/*.png
data/haze4k/test/gt/*.png
```

The dataset is not bundled. `DATA_MANIFEST.json` fixes image identities, order,
source-file hashes, RGB array shapes, and cache offsets. The entry point checks
the source hashes and builds the required split cache. A later call checks the
cache hash before reusing it. Keep the image files in their original encoding.

From the package root:

```text
python paper.py eval --id haze4k-full-seed42 --dry-run
python paper.py eval --suite haze4k --data-root data --output runs/haze-eval
python paper.py collect --suite haze4k --recorded --output runs/haze-recorded
python paper.py train --id haze4k-full-seed42 --data-root data --output runs/haze-train
python paper.py eval --id haze4k-full-seed42 --trained --data-root data --output runs/haze-train
```

The training command runs the complete budget. It uses batch size 2, 256-pixel
crops, FP32 with TF32 disabled, AdamW, and the exact sampler and augmentation
order in `data.py`. `configs/haze4k.json` lists the optimizer, scheduler, and loss.
The full model uses fused AdamW; other conditions use the standard AdamW path,
matching their original runs. Training saves an alternating resume state every 2,000 updates. Rerunning the same
command resumes an unfinished run. Use a new output directory for a new run.

Evaluation uses all 1,000 test pairs, native whole images, reflection padding to
a multiple of four, and cropping back to the original shape. Predictions are
clipped to [0,1] before metrics. RGB PSNR and SSIM are averaged equally over
images. The direct SSIM implementation uses NumPy/OpenCV float64, Gaussian
11-by-11 valid windows, sigma 1.5, and range 255 after scaling both float images
by 255; it does not quantize them or pool the images.

`PLAN.json` and `RUNTIME_MANIFEST.json` record the portable training-source
identity. Checkpoint SHA-256 values are in the package weight manifest. Bundled
`FINAL.json` files retain all per-image results and the hash of the original
evaluation record. Raw training checkpoints and test predictions are not needed
to read those scores. CUDA is required to rerun full inference or training.
