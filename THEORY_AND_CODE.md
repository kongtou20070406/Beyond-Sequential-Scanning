# NSI equations and code map

This guide links the current restoration paper to the release code.
It uses one fixed module input `f`. The learned weights and generated parameters
stay fixed during its `K` propagation steps. The state changes at every step.
The theorem concerns this fixed-input update, not the training process.

## Read the package

1. Run `python paper.py list --suite current-restoration` for active experiments.
2. Read `PAPER_RESULTS.md` for checkpoint choices and recorded metrics.
3. Read `METRIC_PROTOCOL.md` before you compare SSIM values.
4. Run `python tools/check_nsi_equations.py` for a small CPU formula check.
5. Read the paper Method and Appendix A for the analytic proof.
6. Use [formalization/README.md](formalization/README.md) for the Lean proof,
   its exact coverage, and the commands that check all local declarations.

The GoPro and O-HAZE operator is in
[`token_mixers/ours/implementation.py`](token_mixers/ours/implementation.py).
Its base kernel normalization is in
[`tasks/restoration/runtime/clpo2027/model.py`](tasks/restoration/runtime/clpo2027/model.py).
Haze4K carries the matched operator in
[`tasks/restoration/haze4k/runtime/clpo2027/radial_profiled.py`](tasks/restoration/haze4k/runtime/clpo2027/radial_profiled.py).
These task copies have frozen runtime paths. Keep the paths when you load
released weights.

## Fixed values and changing values

- `f` is the projected input. `d_f = B_f f` is the repeated input term.
- The generator computes raw kernel coefficients, scales, and decay from `f`.
  These values stay fixed for this input.
- `M_f` is the normalized nine-offset kernel. Its actual gain is `q_f`.
  The configured limit is `rho = 0.95`. In exact arithmetic, `q_f < rho`.
- `h_0 = 0`. The state `h_k` changes at each step. Its retention coefficient
  is recomputed from `h_(k-1)`.
- `K` is the selected step count. The theorem allows any integer `K >= 1`.
- The readout weights are learned global scalars. The generator does not
  predict them from the current input.

## Match the formulas to the code

| Paper expression | Release implementation |
|---|---|
| Project `x` to `f` | `_drive_with_attention` in the base operator |
| Generate parameters from `f` | `_coefficient_hidden`, `_lightweight_coefficients`, and `_radial_controls` |
| `M_raw = [[a,-b],[b,a]]` | `ar` and `ai` in `_diagonal_coefficients_from_raw` and `_apply_stencil` |
| Normalize nine offsets | `_diagonal_coefficients_from_raw` uses `rho / (1 + sum(sqrt(a*a+b*b)))` |
| `d_f = B_f f` | `write * fr` and `write * fi` in `forward` |
| Retain state | `_select_state` computes one scale per group and position |
| `h_k = L_f(d_f + R_f(h_(k-1)))` | `_state_depth_step` and `_profiled_rollout` |
| `z = sum_k omega_k (h_k-h_(k-1))` | `depth_weights` and `_profiled_rollout` |
| `y = W_out(C_f z)` | `read * profiled_r/i` and `_interface_readout` |

The real and imaginary values form one two-channel pair. For each kernel
matrix, its spectral norm is `sqrt(a*a+b*b)`. The base operator divides all
nine matrices for one pair and receiving position by the same denominator.
This gives `q_f < rho` before floating-point rounding.

For a group of `g` pairs, the code uses
`E = mean_pair(real*real + imag*imag)` and mean decay `bar_lambda_f`.
It computes `eta = (1 + bar_lambda_f E)^(-1/2)`.
The paper averages over `2g` real channels. Thus,
`E = 2 RMS_g^2` and `lambda_f = 2 bar_lambda_f`.
These expressions define the same retention rule. The code applies `eta` at
the sending position before aggregation at the receiving position.

For `K > 1`, `depth_weights` uses `tau_k = (k-1)/(K-1)` and
`omega_k = K softmax_k(theta_1^2 tau_k + theta_2^2 tau_k^2)`.
For `K = 1`, its one-element softmax gives `omega_1 = 1`.
The weights are positive, do not decrease with `k`, and sum to `K`.

## How the proof reaches its result

Use `N(h) = max_(batch,pair,position) ||h_pair||_2` for a fixed input.
Kernel normalization gives `N(L_f v) <= q_f N(v)`.
For one group, retention is `R(v) = v / sqrt(1 + alpha ||v||_2^2)`,
where `alpha = bar_lambda_f/g >= 0`.
Set `s = alpha ||v||_2^2/(1+alpha ||v||_2^2)` and
`u = v/||v||_2` for nonzero `v`. Differentiation gives
`DR(v) = sqrt(1-s) (I - s uu^T)`.
The reflection `H = I - 2uu^T` has induced `N` norm at most `sqrt(g)`.
Because `I - s uu^T = (1-s/2)I + (s/2)H`, for `g <= 4` we obtain
`||DR(v)||_N <= sqrt(1-s) (1+s/2) <= 1`.
The final inequality follows by squaring:
`(1-s)(1+s/2)^2 = 1 - 3s^2/4 - s^3/4 <= 1`.
Integrating this Jacobian bound along a line segment proves that retention
is non-expansive between any two states.
This fact compares retention at two different states. The simpler fact
`N(R(h)) <= N(h)` alone cannot prove contraction.

The complete map `T_f(h) = L_f(d_f + R(h))` therefore obeys
`N(T_f(h)-T_f(v)) <= q_f N(h-v)`.
Since `q_f < 1`, this is a contraction with a unique fixed point.
The same map is used at every step. Comparing consecutive trajectories gives
`N(Delta h_k) <= q_f^(k-1) N(Delta h_1) <= q_f^k N(d_f)`.
With `h_0 = 0`, a geometric sum gives
`N(h_K) <= q_f (1-q_f^K)/(1-q_f) N(d_f)`.

Readout weights increase with `k`, while `q_f^k` decreases. Their opposite
order and `sum_k omega_k = K` give the reverse Chebyshev inequality
`sum_k omega_k q_f^k <= sum_k q_f^k`.
One way to see the sign is that
`K sum_k omega_k q_f^k - (sum_k omega_k)(sum_k q_f^k)` equals
`sum_(i<j) (omega_i-omega_j)(q_f^i-q_f^j) <= 0`.
The triangle inequality gives the same finite-step bound for `N(z)`.
Both bounds are at most `q_f/(1-q_f) N(d_f)` for all `K`.
They apply before output scaling `C_f` and projection `W_out`.

The proof assumes zero values outside the grid, nonnegative decay,
`g <= 4`, fixed input-conditioned parameters, and `q_f < 1`.
It concerns exact arithmetic. The paper appendix states a separate bound for
finite-precision errors. The small CPU check in this package does not prove
the theorem for all inputs.
