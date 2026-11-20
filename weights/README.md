# Restoration weights

Extract this weights ZIP and the code ZIP into the same parent folder.
Both archives use the code/ root. The weights are in code/weights/.

Run `python paper.py verify --suite current-restoration` from code/.
Use `checksums.json` for the exact SHA-256 values and file sizes.
Use `manifest.json` for the selection rules and total file count.
All included tensor files are byte-for-byte copies of the verified source release.

The current paper uses 55 unique endpoint files. Additional restoration files
support the fixed initialization, historical controls, and the MaIR saved-checkpoint
comparison. These files are not additional current paper conditions.
GoPro native-evaluation aliases reuse the main weights.
The five O-HAZE historical controls without direct SSIM are excluded from the
current-restoration suite and from the current result collector.
