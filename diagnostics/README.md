# Fixed-checkpoint diagnostics

These files support the ERF, module cost, state trajectory, and contraction
diagnostics in the current manuscript. No command in this folder trains a model.
The recorded numerical results and array files are retained. Local path metadata
is replaced with relative paths. SOURCE_PROVENANCE.json records both file hashes.
The original checkpoint hashes remain unchanged.

Run commands from the code directory. Install the main requirements first.
Use Python 3.12 and CUDA for image-level probes.

## Read and verify ERF records

```text
python diagnostics/erf_paper_test_20260924/verify_erf.py
python diagnostics/erf_paper_test_20260924/plot_erf.py
```

The saved probe uses one test image and one central output pixel per dataset,
with seed 42. It is not a dataset average or a three-seed estimate. The scalar
target is the mean RGB value of the central output residual (prediction minus
input). Each map sums the absolute input-channel derivatives. r90 is the radius
that encloses 90 percent of that gradient mass. The derivative is with respect
to the preprocessed, unpadded input. Images are not bundled.

To run a new probe, set NSI_DATA_ROOT to a parent folder with ohaze and gopro
folders in the main README layout. Set NSI_HAZE_CACHE to the Haze4K cache built
by the main Haze4K evaluator. It must contain test.bin. Then run, for example:

```text
python diagnostics/erf_paper_test_20260924/run_erf.py ohaze ours --checkpoint-blocks
```

New ERF maps go to runs/erf. Set NSI_ERF_OUTPUT to use another output folder.
The plot and verification scripts read the bundled results folder. To plot a
new complete probe, copy its results into a separate copy of this diagnostics
folder. Spatial-Mamba probes need the official extensions described in
supplement/THIRD_PARTY.md; SUPPLEMENT_SPATIAL_EXTENSIONS selects a local build.

## Count module MACs on CPU

```text
python diagnostics/module_cost_comparison_20260915/measure.py
python diagnostics/module_cost_comparison_20260915/standalone_and_report.py
python diagnostics/token_mixer_method_20260912/profile_module_macs.py --help
```

The SS2D CPU counter preserves projection shapes and adds the official scan
estimate. It does not approximate scan outputs or measure latency.

## Inspect state and gain diagnostics

```text
python diagnostics/gain_state_trajectory_20260916/run.py
python diagnostics/contraction_performance_20260921/analyze_weights.py --phase smoke
python diagnostics/contraction_performance_20260921/analyze_weights.py --phase profile
python diagnostics/contraction_performance_20260921/analyze_weights.py --phase intervene
python diagnostics/contraction_performance_20260921/summarize.py
```

The trajectory script uses NSI_DATA_ROOT. Haze4K diagnostics use NSI_HAZE_CACHE.
These scripts write new measurements beside the bundled records. Use a copy
of this diagnostics folder if you need to keep the original records in place.
Measured directional Jacobian gains are not induced-norm certificates. Refer
to THEORY_AND_CODE.md and the manuscript for the sufficient stability conditions.
