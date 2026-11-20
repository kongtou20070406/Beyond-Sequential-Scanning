import Mathlib

namespace NSI
open scoped BigOperators
open WithLp

abbrev PairState := EuclideanSpace ℝ (Fin 2)

/-- The actual real two-channel transform used by each NSI spatial connection. -/
def pairTransform (a b : ℝ) (v : PairState) : PairState :=
  toLp 2 ![a * v 0 - b * v 1, b * v 0 + a * v 1]

theorem pairTransform_norm_sq (a b : ℝ) (v : PairState) :
    ‖pairTransform a b v‖ ^ 2 = (a ^ 2 + b ^ 2) * ‖v‖ ^ 2 := by
  simp only [EuclideanSpace.norm_sq_eq, Fin.sum_univ_two, Real.norm_eq_abs, sq_abs,
    pairTransform, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one]
  ring

theorem pairTransform_norm (a b : ℝ) (v : PairState) :
    ‖pairTransform a b v‖ = Real.sqrt (a ^ 2 + b ^ 2) * ‖v‖ := by
  have h := pairTransform_norm_sq a b v
  have hs := Real.sq_sqrt (show 0 ≤ a ^ 2 + b ^ 2 by positivity)
  have hp : 0 ≤ Real.sqrt (a ^ 2 + b ^ 2) * ‖v‖ := by positivity
  nlinarith [norm_nonneg (pairTransform a b v)]

theorem pairTransform_sub (a b : ℝ) (v w : PairState) :
    pairTransform a b (v-w) = pairTransform a b v - pairTransform a b w := by
  ext i
  fin_cases i <;> simp [pairTransform, mul_sub] <;> ring

theorem pairTransform_smul (a b r : ℝ) (v : PairState) :
    pairTransform a b (r • v) = r • pairTransform a b v := by
  ext i
  fin_cases i <;> simp [pairTransform, smul_eq_mul] <;> ring

variable {P : Type*} [Fintype P]

/-- `none` denotes a neighbor outside the finite grid. -/
def zeroBoundaryRead (h : P → PairState) : Option P → PairState
  | none => 0
  | some p => h p

theorem zeroBoundaryRead_norm_le (h : P → PairState) (p : Option P) :
    ‖zeroBoundaryRead h p‖ ≤ ‖h‖ := by
  cases p with
  | none => simp [zeroBoundaryRead]
  | some p => exact norm_le_pi_norm h p

def neighborhoodAggregate (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (h : P → PairState) : P → PairState :=
  fun p => ∑ δ, pairTransform (a p δ) (b p δ) (zeroBoundaryRead h (neighbor p δ))

theorem neighborhoodAggregate_point_bound (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (h : P → PairState) (p : P) :
    ‖neighborhoodAggregate a b neighbor h p‖ ≤
      (∑ δ, Real.sqrt ((a p δ)^2 + (b p δ)^2)) * ‖h‖ := by
  calc
    _ ≤ ∑ δ, ‖pairTransform (a p δ) (b p δ)
        (zeroBoundaryRead h (neighbor p δ))‖ := norm_sum_le _ _
    _ ≤ ∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2) * ‖h‖ := by
      apply Finset.sum_le_sum
      intro δ _
      rw [pairTransform_norm]
      exact mul_le_mul_of_nonneg_left (zeroBoundaryRead_norm_le h _)
        (Real.sqrt_nonneg _)
    _ = _ := (Finset.sum_mul _ _ _).symm

theorem neighborhoodAggregate_norm_bound (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (q : ℝ) (hq : 0 ≤ q)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (h : P → PairState) :
    ‖neighborhoodAggregate a b neighbor h‖ ≤ q * ‖h‖ := by
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg hq (norm_nonneg h))).2
  intro p
  exact (neighborhoodAggregate_point_bound a b neighbor h p).trans
    (mul_le_mul_of_nonneg_right (hgain p) (norm_nonneg h))

theorem normalized_gain_lt (ρ m : ℝ) (hρ : 0 < ρ) (hm : 0 ≤ m) :
    ρ * m / (1 + m) < ρ := by
  apply (div_lt_iff₀ (by linarith : 0 < 1 + m)).2
  nlinarith

omit [Fintype P] in
theorem zeroBoundaryRead_sub (h v : P → PairState) (p : Option P) :
    zeroBoundaryRead (h-v) p = zeroBoundaryRead h p - zeroBoundaryRead v p := by
  cases p <;> simp [zeroBoundaryRead]

omit [Fintype P] in
theorem neighborhoodAggregate_sub (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (h v : P → PairState) :
    neighborhoodAggregate a b neighbor (h-v) =
      neighborhoodAggregate a b neighbor h - neighborhoodAggregate a b neighbor v := by
  ext p i
  simp only [neighborhoodAggregate, zeroBoundaryRead_sub, pairTransform_sub,
    Finset.sum_sub_distrib, Pi.sub_apply]

theorem neighborhoodAggregate_lipschitz (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q) :
    LipschitzWith q (neighborhoodAggregate a b neighbor) := by
  apply LipschitzWith.of_dist_le_mul
  intro h v
  simpa only [dist_eq_norm, neighborhoodAggregate_sub] using
    neighborhoodAggregate_norm_bound a b neighbor q q.coe_nonneg hgain (h-v)

omit [Fintype P] in
theorem neighborhoodAggregate_smul (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (r : ℝ) (h : P → PairState) :
    neighborhoodAggregate a b neighbor (r • h) = r • neighborhoodAggregate a b neighbor h := by
  have hz (p : Option P) : zeroBoundaryRead (r • h) p = r • zeroBoundaryRead h p := by
    cases p <;> simp [zeroBoundaryRead]
  funext p
  simp only [neighborhoodAggregate, hz, pairTransform_smul, Pi.smul_apply,
    Finset.smul_sum]

/-- Identity retention has no uniform old-state contribution bound for a nonzero spatial map. -/
theorem identity_old_contribution_unbounded (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (d v : P → PairState)
    (hv : neighborhoodAggregate a b neighbor v ≠ 0) (B : ℝ) :
    ∃ h, B < ‖neighborhoodAggregate a b neighbor (d+h) -
      neighborhoodAggregate a b neighbor d‖ := by
  let r := (|B|+1) / ‖neighborhoodAggregate a b neighbor v‖
  have hn : 0 < ‖neighborhoodAggregate a b neighbor v‖ := norm_pos_iff.mpr hv
  have hr : 0 ≤ r := by dsimp [r]; positivity
  refine ⟨r • v, ?_⟩
  rw [← neighborhoodAggregate_sub]
  rw [show d + r • v - d = r • v by abel]
  rw [neighborhoodAggregate_smul, norm_smul, Real.norm_eq_abs, abs_of_nonneg hr]
  dsimp [r]
  rw [div_mul_cancel₀ _ hn.ne']
  linarith [le_abs_self B]

theorem scaled_pair_gain (r a b : ℝ) (hr : 0 ≤ r) :
    Real.sqrt ((r*a)^2+(r*b)^2) = r * Real.sqrt (a^2+b^2) := by
  rw [show (r*a)^2+(r*b)^2 = r^2*(a^2+b^2) by ring,
    Real.sqrt_mul (sq_nonneg r), Real.sqrt_sq_eq_abs, abs_of_nonneg hr]

theorem normalized_nine_gains (a b : Fin 9 → ℝ) (ρ : ℝ) (hρ : 0 < ρ) :
    let m := ∑ δ, Real.sqrt ((a δ)^2+(b δ)^2)
    (∑ δ, Real.sqrt (((ρ/(1+m))*a δ)^2+((ρ/(1+m))*b δ)^2)) < ρ := by
  intro m
  have hm : 0 ≤ m := Finset.sum_nonneg (fun _ _ => Real.sqrt_nonneg _)
  have ht : 0 ≤ ρ / (1+m) := div_nonneg hρ.le (by linarith)
  simp_rw [scaled_pair_gain _ _ _ ht]
  rw [← Finset.mul_sum]
  change ρ / (1+m) * m < ρ
  rw [div_mul_eq_mul_div]
  exact normalized_gain_lt ρ m hρ hm

noncomputable def normalizeKernel (a b : P → Fin 9 → ℝ) (ρ : ℝ) :
    P → Fin 9 → ℝ :=
  fun p δ => ρ / (1 + ∑ j, Real.sqrt ((a p j)^2+(b p j)^2)) * a p δ

omit [Fintype P] in
theorem normalizedKernel_gain (a b : P → Fin 9 → ℝ) (ρ : ℝ) (hρ : 0 < ρ)
    (p : P) :
    (∑ δ, Real.sqrt ((normalizeKernel a b ρ p δ)^2+
      (normalizeKernel b a ρ p δ)^2)) < ρ := by
  simpa only [normalizeKernel, add_comm ((b p _ )^2) ((a p _ )^2)] using
    normalized_nine_gains (a p) (b p) ρ hρ

theorem normalizedKernel_lipschitz (a b : P → Fin 9 → ℝ)
    (neighbor : P → Fin 9 → Option P) (ρ : NNReal) (hρ : 0 < (ρ : ℝ)) :
    LipschitzWith ρ
      (neighborhoodAggregate (normalizeKernel a b ρ) (normalizeKernel b a ρ) neighbor) :=
  neighborhoodAggregate_lipschitz _ _ _ ρ
    (fun p => (normalizedKernel_gain a b ρ hρ p).le)

#print axioms pairTransform_norm
#print axioms neighborhoodAggregate_norm_bound
#print axioms normalized_gain_lt
#print axioms neighborhoodAggregate_lipschitz
#print axioms normalized_nine_gains
#print axioms normalizedKernel_lipschitz
#print axioms identity_old_contribution_unbounded
end NSI
