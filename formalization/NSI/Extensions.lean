import Mathlib

namespace NSI

open Finset Function Real

/-! Finite-precision accumulation under one shared contraction. -/

/-- A general finite-precision error bound, represented by its exact recursive convolution. -/
def precisionBound (q : ℝ) (ε : ℕ → ℝ) (e₀ : ℝ) : ℕ → ℝ
  | 0 => e₀
  | k + 1 => q * precisionBound q ε e₀ k + ε k

theorem finite_precision_le_precisionBound
    (q : ℝ) (ε e : ℕ → ℝ) (e₀ : ℝ) (hq : 0 ≤ q)
    (hzero : e 0 ≤ e₀)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + ε k) :
    ∀ k, e k ≤ precisionBound q ε e₀ k := by
  intro k
  induction k with
  | zero => exact hzero
  | succ k ih =>
      refine (hstep k).trans ?_
      simpa [precisionBound] using
        (add_le_add_right (mul_le_mul_of_nonneg_left ih hq) (ε k))

/-- With a common local error and equal initial states, the accumulated error is a finite geometric sum. -/
theorem finite_precision_constant_error
    (q ε : ℝ) (e : ℕ → ℝ) (hq : 0 ≤ q) (_hε : 0 ≤ ε)
    (hzero : e 0 ≤ 0)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + ε) :
    ∀ k, e k ≤ ε * ∑ i ∈ range k, q ^ i := by
  intro k
  induction k with
  | zero => simpa using hzero
  | succ k ih =>
      calc
        e (k + 1) ≤ q * e k + ε := hstep k
        _ ≤ q * (ε * ∑ i ∈ range k, q ^ i) + ε := by gcongr
        _ = ε * ∑ i ∈ range (k + 1), q ^ i := by
          rw [sum_range_succ]
          linear_combination ε * (geom_sum_mul q k)

/-- Closed finite-precision bound for a common per-step error. -/
theorem finite_precision_constant_error_closed
    (q ε : ℝ) (e : ℕ → ℝ) (hq : 0 ≤ q) (hq1 : q < 1) (hε : 0 ≤ ε)
    (hzero : e 0 ≤ 0)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + ε) (k : ℕ) :
    e k ≤ ε * ((1 - q ^ k) / (1 - q)) := by
  refine (finite_precision_constant_error q ε e hq hε hzero hstep k).trans_eq ?_
  have hne : 1 - q ≠ 0 := (sub_pos.mpr hq1).ne'
  rw [show (∑ i ∈ range k, q ^ i) = (1 - q ^ k) / (1 - q) by
    apply (eq_div_iff hne).2
    exact geom_sum_mul_neg q k]

/-- Uniform `ε/(1-q)` finite-precision bound. -/
theorem finite_precision_constant_error_uniform
    (q ε : ℝ) (e : ℕ → ℝ) (hq : 0 ≤ q) (hq1 : q < 1) (hε : 0 ≤ ε)
    (hzero : e 0 ≤ 0)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + ε) (k : ℕ) :
    e k ≤ ε / (1 - q) := by
  calc
    e k ≤ ε * ((1 - q ^ k) / (1 - q)) :=
      finite_precision_constant_error_closed q ε e hq hq1 hε hzero hstep k
    _ ≤ ε / (1 - q) := by
      have hden : 0 < 1 - q := sub_pos.mpr hq1
      have hpow : 0 ≤ q ^ k := pow_nonneg hq k
      have hnum : 1 - q ^ k ≤ 1 := by linarith
      exact (mul_le_mul_of_nonneg_left
        (div_le_div_of_nonneg_right hnum hden.le) hε).trans_eq (by ring)

/-! Exact counterexamples for step-dependent maps and state-dependent kernels. -/

noncomputable def alternatingPlus (x : ℝ) : ℝ := (1 + x) / 2
noncomputable def alternatingMinus (x : ℝ) : ℝ := -(1 + x) / 2

theorem alternatingPlus_lipschitz : LipschitzWith (1 / 2 : NNReal) alternatingPlus := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simp only [alternatingPlus, Real.dist_eq, NNReal.coe_div, NNReal.coe_one,
    NNReal.coe_ofNat]
  rw [show (1 + x) / 2 - (1 + y) / 2 = (x - y) / 2 by ring]
  norm_num [abs_div, div_eq_mul_inv]
  rw [mul_comm]

theorem alternatingMinus_lipschitz : LipschitzWith (1 / 2 : NNReal) alternatingMinus := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simp only [alternatingMinus, Real.dist_eq, NNReal.coe_div, NNReal.coe_one,
    NNReal.coe_ofNat]
  rw [show -(1 + x) / 2 - (-(1 + y) / 2) = -(x - y) / 2 by ring]
  rw [abs_div, abs_neg, abs_sub_comm]
  norm_num [div_eq_mul_inv]
  rw [mul_comm]

theorem alternating_two_cycle :
    alternatingPlus (-3 / 5) = (1 / 5 : ℝ) ∧
    alternatingMinus (1 / 5) = (-3 / 5 : ℝ) := by
  constructor <;> norm_num [alternatingPlus, alternatingMinus]

theorem alternating_two_cycle_increment_nonzero :
    |(1 / 5 : ℝ) - (-3 / 5)| = 4 / 5 := by norm_num

noncomputable def scalarRetention (x : ℝ) : ℝ := x / sqrt (1 + x ^ 2)
noncomputable def stateKernel (x : ℝ) : ℝ := (19 / 20) * tanh (2 * x)
noncomputable def stateKernelUpdate (x : ℝ) : ℝ := stateKernel x * (1 + scalarRetention x)

lemma scalarRetention_differentiableAt_zero : DifferentiableAt ℝ scalarRetention 0 := by
  unfold scalarRetention
  have hid : HasDerivAt (fun x : ℝ ↦ x) 1 0 := hasDerivAt_id 0
  have hinner : HasDerivAt (fun x : ℝ ↦ 1 + x ^ 2) 0 0 := by
    have hp := hid.mul hid
    change HasDerivAt (fun x : ℝ ↦ x * x) _ 0 at hp
    norm_num at hp
    simpa only [pow_two] using HasDerivAt.const_add (1 : ℝ) hp
  have hsqrt : HasDerivAt (fun x : ℝ ↦ sqrt (1 + x ^ 2)) 0 0 := by
    convert hinner.sqrt (by norm_num) using 1; norm_num
  exact (hid.div hsqrt (by norm_num)).differentiableAt

theorem stateKernel_hasDerivAt_zero : HasDerivAt stateKernel (19 / 10) 0 := by
  have htanh : HasDerivAt (fun x : ℝ ↦ tanh x) 1 0 := by
    have h := (Real.hasDerivAt_sinh 0).div (Real.hasDerivAt_cosh 0) (by norm_num)
    change HasDerivAt (fun x : ℝ ↦ sinh x / cosh x) _ 0 at h
    norm_num at h
    simpa only [Real.tanh_eq_sinh_div_cosh] using h
  have htwo : HasDerivAt (fun x : ℝ ↦ 2 * x) 2 0 := by
    have h := (hasDerivAt_const (x := 0) (2 : ℝ)).mul (hasDerivAt_id 0)
    change HasDerivAt (fun x : ℝ ↦ 2 * x) _ 0 at h
    norm_num at h
    exact h
  have hcomp : HasDerivAt (fun x : ℝ ↦ tanh (2 * x)) 2 0 := by
    have h := htanh.comp_of_eq 0 htwo (by norm_num)
    change HasDerivAt (fun x : ℝ ↦ tanh (2 * x)) _ 0 at h
    norm_num at h
    exact h
  unfold stateKernel
  have h := hcomp.const_mul (19 / 20)
  norm_num at h
  exact h

theorem stateKernel_abs_lt (x : ℝ) : |stateKernel x| < 19 / 20 := by
  rw [stateKernel, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 19 / 20)]
  nlinarith [Real.abs_tanh_lt_one (2 * x)]

theorem stateKernelUpdate_hasDerivAt_zero :
    HasDerivAt stateKernelUpdate (19 / 10) 0 := by
  have hL := stateKernel_hasDerivAt_zero
  have hR := scalarRetention_differentiableAt_zero
  have hsum : HasDerivAt (fun x : ℝ ↦ 1 + scalarRetention x)
      (deriv scalarRetention 0) 0 := by
    exact HasDerivAt.const_add 1 hR.hasDerivAt
  have hprod : HasDerivAt
      (fun x : ℝ ↦ stateKernel x * (1 + scalarRetention x)) (19 / 10) 0 := by
    have h := hL.mul hsum
    change HasDerivAt (fun x : ℝ ↦ stateKernel x * (1 + scalarRetention x)) _ 0 at h
    norm_num [stateKernel, scalarRetention] at h
    exact h
  norm_num [stateKernel, scalarRetention] at hprod
  unfold stateKernelUpdate
  exact hprod

theorem stateKernelUpdate_deriv_zero : deriv stateKernelUpdate 0 = 19 / 10 :=
  stateKernelUpdate_hasDerivAt_zero.deriv

theorem stateKernelUpdate_not_one_lipschitz :
    ¬ LipschitzWith 1 stateKernelUpdate := by
  intro h
  have hd := norm_deriv_le_of_lipschitz (x₀ := (0 : ℝ)) h
  rw [stateKernelUpdate_deriv_zero] at hd
  norm_num at hd

/-! Positive-retention scalar envelope. -/

noncomputable def positiveRetention (α r : ℝ) : ℝ :=
  r / sqrt (1 + α * r ^ 2)

lemma positiveRetention_den_pos
    {α r : ℝ} (hα : 0 ≤ α) : 0 < sqrt (1 + α * r ^ 2) := by
  apply Real.sqrt_pos.mpr
  nlinarith [mul_nonneg hα (sq_nonneg r)]

/-- Positive retention never exceeds the old nonnegative magnitude. -/
theorem positiveRetention_le_self
    {α r : ℝ} (hα : 0 ≤ α) (hr : 0 ≤ r) :
    positiveRetention α r ≤ r := by
  unfold positiveRetention
  have hden1 : 1 ≤ sqrt (1 + α * r ^ 2) := by
    rw [Real.le_sqrt (by norm_num) (by nlinarith [mul_nonneg hα (sq_nonneg r)])]
    nlinarith [mul_nonneg hα (sq_nonneg r)]
  exact div_le_self hr hden1

/-- Strict decay away from zero when the retention strength is positive. -/
theorem positiveRetention_lt_self
    {α r : ℝ} (hα : 0 < α) (hr : 0 < r) :
    positiveRetention α r < r := by
  unfold positiveRetention
  have hden1 : 1 < sqrt (1 + α * r ^ 2) := by
    rw [Real.lt_sqrt (by norm_num)]
    nlinarith [mul_pos hα (sq_pos_of_pos hr)]
  exact (div_lt_iff₀ (positiveRetention_den_pos hα.le)).2 (by nlinarith)

/-- The positive-retention contribution is uniformly bounded by `1/sqrt α`. -/
theorem positiveRetention_le_inv_sqrt
    {α r : ℝ} (hα : 0 < α) (hr : 0 ≤ r) :
    positiveRetention α r ≤ 1 / sqrt α := by
  unfold positiveRetention
  have haS : 0 < sqrt α := Real.sqrt_pos.mpr hα
  have hiS : 0 < sqrt (1 + α * r ^ 2) := positiveRetention_den_pos hα.le
  apply (div_le_div_iff₀ hiS haS).2
  have hsqa : (sqrt α) ^ 2 = α := Real.sq_sqrt hα.le
  have hsqi : (sqrt (1 + α * r ^ 2)) ^ 2 = 1 + α * r ^ 2 :=
    Real.sq_sqrt (by nlinarith [mul_nonneg hα.le (sq_nonneg r)])
  have hleft : 0 ≤ r * sqrt α := mul_nonneg hr haS.le
  have hright : 0 ≤ sqrt (1 + α * r ^ 2) := hiS.le
  simp only [one_mul]
  rw [← sq_le_sq₀ hleft hright]
  rw [mul_pow, hsqa, hsqi]
  nlinarith

/-- Any positive fixed point of the nonlinear envelope lies strictly below the identity-retention bound. -/
theorem positiveRetention_fixedPoint_lt_geometric
    {q d α B : ℝ} (hq : 0 < q) (hq1 : q < 1) (hd : 0 < d) (hα : 0 < α)
    (hB0 : 0 ≤ B)
    (hfix : B = q * (d + positiveRetention α B)) :
    0 < B ∧ B < q * d / (1 - q) := by
  have hphi0 : 0 ≤ positiveRetention α B := by
    unfold positiveRetention
    positivity
  have hB : 0 < B := by nlinarith [mul_pos hq hd]
  have hstrict := positiveRetention_lt_self hα hB
  have hsub : 0 < 1 - q := sub_pos.mpr hq1
  constructor
  · exact hB
  · have hineq : B < q * (d + B) := by
      calc
        B = q * (d + positiveRetention α B) := hfix
        _ < q * (d + B) := by
          exact mul_lt_mul_of_pos_left (by linarith) hq
    apply (lt_div_iff₀ hsub).2
    nlinarith

/-- Abstract old-state contribution bound used after the scalar retention envelope. -/
theorem bounded_old_state_contribution
    {E F : Type*} [SeminormedAddCommGroup E] [SeminormedAddCommGroup F]
    [NormedSpace ℝ E] [NormedSpace ℝ F]
    (q : NNReal) (L : E → F) (R : E → E) (hL : LipschitzWith q L)
    (S : ℝ) (_hS : 0 ≤ S) (hR : ∀ x, ‖R x‖ ≤ S) (x : E) :
    ‖L (R x) - L 0‖ ≤ (q : ℝ) * S := by
  calc
    ‖L (R x) - L 0‖ = dist (L (R x)) (L 0) := by rw [dist_eq_norm]
    _ ≤ (q : ℝ) * dist (R x) 0 := hL.dist_le_mul _ _
    _ = (q : ℝ) * ‖R x‖ := by rw [dist_zero_right]
    _ ≤ (q : ℝ) * S := mul_le_mul_of_nonneg_left (hR x) q.coe_nonneg

#print axioms finite_precision_le_precisionBound
#print axioms finite_precision_constant_error
#print axioms finite_precision_constant_error_closed
#print axioms finite_precision_constant_error_uniform
#print axioms alternatingPlus_lipschitz
#print axioms alternatingMinus_lipschitz
#print axioms alternating_two_cycle
#print axioms alternating_two_cycle_increment_nonzero
#print axioms stateKernelUpdate_hasDerivAt_zero
#print axioms stateKernelUpdate_deriv_zero
#print axioms stateKernel_abs_lt
#print axioms stateKernelUpdate_not_one_lipschitz
#print axioms positiveRetention_le_self
#print axioms positiveRetention_lt_self
#print axioms positiveRetention_le_inv_sqrt
#print axioms positiveRetention_fixedPoint_lt_geometric
#print axioms bounded_old_state_contribution

end NSI
