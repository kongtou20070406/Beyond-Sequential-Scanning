# Lean proof of NSI stability

This directory contains the Lean source for the fixed-input NSI update.
It proves the actual nonlinear retention map and the finite-grid update.
The result is conditional on the assumptions below, as in the paper.

## Run the check

Use Python 3.11 or later, Git, and Lean through elan.
The files `lean-toolchain` and `lake-manifest.json` fix the versions:

- Lean: `leanprover/lean4:v4.32.2`.
- mathlib: `905b95818eb32af7874a58b427f50c1711a5e96c`.

From this directory, run:

```text
lake exe cache get
python check_all.py
```

The first command downloads the pinned dependencies and their build cache.
The second command rebuilds every local proof file in a new build directory.
It treats warnings as errors. It then checks the transitive axiom dependencies
of every declaration from each local proof module, including private helpers.
It accepts only `propext`, `Classical.choice`, and `Quot.sound`, which are the
standard logical axioms used by this Lean/mathlib development.
Unproved placeholders, extra axioms, and unsafe proof declarations fail the check.

For an existing checkout at the exact mathlib commit, set `NSI_MATHLIB` to that
checkout. `NSI_LEAN` can select the Lean executable. Both are optional.
The checker verifies the Lean version and mathlib commit before compilation.
Use the supplied dependency lock. Do not update to a different mathlib commit.

The `verification/` directory contains the compiler logs and `report.json`
from the release check. The report records the source hashes and the axiom
dependencies. A new check replaces these records. `.lake/` contains generated
build files and is not part of the release.

The release check passed for 19 Lean files and 254 local declarations,
including 203 theorems, with no compiler warnings. Separate invalid controls
confirmed that the audit rejects an extra axiom and an unproved placeholder.
These invalid control sources are not included in this package.

## Match the mathematical objects

`GridState P g` is a finite function from `P x Fin g` to real two-channel pairs.
`P` combines sample, group, and position indices. Its norm is the maximum
Euclidean pair norm. `neighbor` supplies nine indices; `none` gives a zero
value outside the grid. The theorem permits any such finite neighbor map,
including the paper's two-dimensional stencil.

For one fixed input, the raw kernels, repeated input `d`, and decay `alpha`
stay fixed. The update is exactly

```text
gridUpdate h = neighborhoodAggregate (d + gridRetention h)
radialRetention alpha x = x / sqrt (1 + alpha * ||x||_2^2)
```

The paper uses `alpha = lambda_f/(2*g) = bar_lambda_f/g`.
The retention coefficient depends on the current state. Its derivative is
included in the proof; it is not treated as a constant coefficient.

## Paper-to-theorem map

| Paper result | Lean declaration in namespace `NSI` |
|---|---|
| Pair transform norm and normalized kernel gain | `pairTransform_norm`, `normalizedKernel_gain` |
| Complete derivative of retention, including the zero state | `hasFDerivAt_radialRetention` |
| Retention is non-expansive for `1 <= g <= 4` | `radialRetention_blockSup_nonexpansive` |
| Complete nonlinear update is Lipschitz with constant `q` | `gridUpdate_lipschitz` |
| Two trajectories contract and the fixed point is unique | `grid_trajectories_contract`, `grid_unique_fixedPoint` |
| Finite and uniform state bounds | `gridState_finite_bound`, `gridState_uniform_bound` |
| Decay of successive increments of one shared map | `shared_map_increment_decay` |
| Finite and uniform incremental readout bounds | `grid_readout_finite`, `grid_readout_uniform` |
| Equivalent weighted state expression | `weighted_difference_identity`, `equal_weight_readout` |
| Bounded retained contribution under positive decay | `gridUpdate_old_contribution_bound`, `gridUpdate_saturation_bound` |
| Scalar state envelope and its fixed point | `gridState_scalar_envelope`, `scalarEnvelope_positive_fixedPoint` |
| Error bound for an inexact orbit | `inexact_orbit_bound` |
| Counterexamples for step-dependent maps and kernels | `alternating_two_cycle`, `stateKernelUpdate_not_one_lipschitz` |
| Expansion for every group size of at least five, at `alpha = 1` | `actual_all_groups_deriv_expands`, `actualRsupAll_not_one_lipschitz` |

The state bound needs nonnegative decay and `0 <= q < 1`; it holds for every
group size. Contraction and the stated readout bound also need `1 <= g <= 4`.
Readout starts at zero and uses nonnegative, nondecreasing weights with sum `K`.
The formal readout theorem takes these weight properties as hypotheses.
It applies to the weights in the paper; the softmax implementation itself is
not a separate theorem in this source. The stronger analytic sharpness
argument for arbitrary positive decay remains in the paper appendix; the
Lean counterexample fixes `alpha = 1` for every group size of at least five.

These results concern exact real arithmetic before output scaling and
projection. The error theorem assumes a bound on each local error; it does
not derive GPU rounding errors. This is not a formal verification of the
PyTorch program, training, or restoration metrics. See
[`../THEORY_AND_CODE.md`](../THEORY_AND_CODE.md) for the implementation map.
