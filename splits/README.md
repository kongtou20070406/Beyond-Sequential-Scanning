# Data splits

Use `reproduce.py --action prepare` to prepare the data.

- O-HAZE uses IDs 1 to 40 for training and IDs 41 to 45 for testing.
- GoPro uses the official training and test sequences.
- Cityscapes uses the official training and validation sets.
- ImageNet-100 uses the class and image lists in `imagenet100`.

Do not change the lists to correct a dataset error.
Correct the source dataset instead.
