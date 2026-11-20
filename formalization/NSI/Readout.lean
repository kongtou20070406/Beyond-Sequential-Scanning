import Mathlib

namespace NSI

open Finset Function

/-! Scalar state-magnitude recurrence. -/

/-- Finite geometric bound for a zero-initialized scalar recurrence. -/
theorem state_recurrence_finite
    (q d : ℝ) (a : ℕ → ℝ) (hq : 0 ≤ q) (_hd : 0 ≤ d)
    (ha0 : a 0 ≤ 0)
    (hstep : ∀ k, a (k + 1) ≤ q * (d + a k)) :
    ∀ k, a k ≤ d * q * ∑ i ∈ range k, q ^ i := by
  intro k
  induction k with
  | zero => simpa using ha0
  | succ k ih =>
      calc
        a (k + 1) ≤ q * (d + a k) := hstep k
        _ ≤ q * (d + d * q * ∑ i ∈ range k, q ^ i) := by gcongr
        _ = d * q * ∑ i ∈ range (k + 1), q ^ i := by
          rw [sum_range_succ]
          linear_combination d * q * (geom_sum_mul q k)

/-- The shifted finite geometric sum in the form used by all NSI bounds. -/
lemma shifted_geom_sum (q : ℝ) (hq1 : q < 1) (k : ℕ) :
    (∑ i ∈ range k, q ^ (i + 1)) = q * (1 - q ^ k) / (1 - q) := by
  have hne : 1 - q ≠ 0 := (sub_pos.mpr hq1).ne'
  apply (eq_div_iff hne).2
  rw [show (∑ i ∈ range k, q ^ (i + 1)) = q * ∑ i ∈ range k, q ^ i by
    simp_rw [pow_succ']; rw [mul_sum]]
  rw [mul_assoc, geom_sum_mul_neg]

/-- Closed finite-geometric form of the scalar recurrence bound. -/
theorem state_recurrence_closed
    (q d : ℝ) (a : ℕ → ℝ) (hq : 0 ≤ q) (hq1 : q < 1) (hd : 0 ≤ d)
    (ha0 : a 0 ≤ 0)
    (hstep : ∀ k, a (k + 1) ≤ q * (d + a k)) (k : ℕ) :
    a k ≤ d * (q * (1 - q ^ k) / (1 - q)) := by
  refine (state_recurrence_finite q d a hq hd ha0 hstep k).trans_eq ?_
  have hne : 1 - q ≠ 0 := (sub_pos.mpr hq1).ne'
  rw [show (∑ i ∈ range k, q ^ i) = (1 - q ^ k) / (1 - q) by
    apply (eq_div_iff hne).2
    exact geom_sum_mul_neg q k]
  ring

/-- Uniform-in-depth consequence of the scalar recurrence. -/
theorem state_recurrence_uniform
    (q d : ℝ) (a : ℕ → ℝ) (hq : 0 ≤ q) (hq1 : q < 1) (hd : 0 ≤ d)
    (ha0 : a 0 ≤ 0)
    (hstep : ∀ k, a (k + 1) ≤ q * (d + a k)) (k : ℕ) :
    a k ≤ d * (q / (1 - q)) := by
  calc
    a k ≤ d * (q * (1 - q ^ k) / (1 - q)) :=
      state_recurrence_closed q d a hq hq1 hd ha0 hstep k
    _ ≤ d * (q / (1 - q)) := by
      have hden : 0 < 1 - q := sub_pos.mpr hq1
      have hpow : 0 ≤ q ^ k := pow_nonneg hq k
      have hnum : q * (1 - q ^ k) ≤ q := by nlinarith [mul_nonneg hq hpow]
      exact mul_le_mul_of_nonneg_left
        (div_le_div_of_nonneg_right hnum hden.le) hd

/-! Decay of state increments under one shared update map. -/

/-- Consecutive increments of the orbit of one shared Lipschitz map decay geometrically. -/
theorem shared_map_increment_decay
    {X : Type*} [PseudoMetricSpace X]
    (q : NNReal) (T : X → X) (hT : LipschitzWith q T)
    (x₀ : X) (d : ℝ) (_hd : 0 ≤ d)
    (hfirst : dist (T x₀) x₀ ≤ (q : ℝ) * d) (k : ℕ) :
    dist ((T^[k + 1]) x₀) ((T^[k]) x₀) ≤ (q : ℝ) ^ (k + 1) * d := by
  induction k with
  | zero => simpa using hfirst
  | succ k ih =>
      calc
        dist ((T^[k + 1 + 1]) x₀) ((T^[k + 1]) x₀)
            = dist (T ((T^[k + 1]) x₀)) (T ((T^[k]) x₀)) := by
              simp only [iterate_succ_apply']
        _ ≤ (q : ℝ) * dist ((T^[k + 1]) x₀) ((T^[k]) x₀) :=
          hT.dist_le_mul _ _
        _ ≤ (q : ℝ) * ((q : ℝ) ^ (k + 1) * d) := by gcongr
        _ = (q : ℝ) ^ (k + 1 + 1) * d := by rw [pow_succ]; ring

/-! Reverse Chebyshev and the weighted incremental readout. -/

/-- Reverse Chebyshev for nondecreasing weights and the decreasing geometric sequence. -/
theorem weighted_geometric_le_unweighted
    (K : ℕ) (q : ℝ) (w : Fin K → ℝ)
    (hq : 0 ≤ q) (hq1 : q ≤ 1)
    (hw : Monotone w) (hsum : ∑ k, w k = K) :
    ∑ k, w k * q ^ (k.val + 1) ≤ ∑ k : Fin K, q ^ (k.val + 1) := by
  by_cases hK : K = 0
  · subst K
    simp
  · have hpow : Antitone (fun k : Fin K ↦ q ^ (k.val + 1)) := by
      intro i j hij
      exact pow_le_pow_of_le_one hq hq1
        (Nat.add_le_add_right (Fin.le_iff_val_le_val.mp hij) 1)
    have hcheb := (hw.antivary hpow).card_mul_sum_le_sum_mul_sum
    simp only [Fintype.card_fin, hsum] at hcheb
    have hKpos : (0 : ℝ) < K := by exact_mod_cast (Nat.pos_of_ne_zero hK)
    nlinarith

/-- The norm of a nonnegatively weighted sum is controlled by weighted norm bounds. -/
theorem norm_weighted_sum_le
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    {K : ℕ} (w : Fin K → ℝ) (v : Fin K → E)
    (hw0 : ∀ k, 0 ≤ w k) (b : Fin K → ℝ)
    (hv : ∀ k, ‖v k‖ ≤ b k) :
    ‖∑ k, w k • v k‖ ≤ ∑ k, w k * b k := by
  calc
    ‖∑ k, w k • v k‖ ≤ ∑ k, ‖w k • v k‖ := norm_sum_le _ _
    _ = ∑ k, w k * ‖v k‖ := by
      apply sum_congr rfl
      intro k _
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw0 k)]
    _ ≤ ∑ k, w k * b k := by
      apply sum_le_sum
      intro k _
      exact mul_le_mul_of_nonneg_left (hv k) (hw0 k)

/-- Full finite weighted-increment readout bound. -/
theorem weighted_increment_readout_bound
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (K : ℕ) (q d : ℝ) (w : Fin K → ℝ) (v : Fin K → E)
    (hq : 0 ≤ q) (hq1 : q < 1) (hd : 0 ≤ d)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w)
    (hsum : ∑ k, w k = K)
    (hv : ∀ k, ‖v k‖ ≤ q ^ (k.val + 1) * d) :
    ‖∑ k, w k • v k‖ ≤ d * (q * (1 - q ^ K) / (1 - q)) := by
  calc
    ‖∑ k, w k • v k‖
        ≤ ∑ k, w k * (q ^ (k.val + 1) * d) :=
          norm_weighted_sum_le w v hw0 _ hv
    _ = d * ∑ k, w k * q ^ (k.val + 1) := by
      simp_rw [← mul_assoc]
      rw [← sum_mul]
      ring
    _ ≤ d * ∑ k : Fin K, q ^ (k.val + 1) := by
      exact mul_le_mul_of_nonneg_left
        (weighted_geometric_le_unweighted K q w hq hq1.le hw hsum) hd
    _ = d * (q * (1 - q ^ K) / (1 - q)) := by
      congr 1
      rw [Fin.sum_univ_eq_sum_range (fun i ↦ q ^ (i + 1)) K]
      exact shifted_geom_sum q hq1 K

/-- Uniform-in-K weighted-increment readout bound. -/
theorem weighted_increment_readout_uniform
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace ℝ E]
    (K : ℕ) (q d : ℝ) (w : Fin K → ℝ) (v : Fin K → E)
    (hq : 0 ≤ q) (hq1 : q < 1) (hd : 0 ≤ d)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w)
    (hsum : ∑ k, w k = K)
    (hv : ∀ k, ‖v k‖ ≤ q ^ (k.val + 1) * d) :
    ‖∑ k, w k • v k‖ ≤ d * (q / (1 - q)) := by
  calc
    ‖∑ k, w k • v k‖ ≤ d * (q * (1 - q ^ K) / (1 - q)) :=
      weighted_increment_readout_bound K q d w v hq hq1 hd hw0 hw hsum hv
    _ ≤ d * (q / (1 - q)) := by
      have hden : 0 < 1 - q := sub_pos.mpr hq1
      have hpow : 0 ≤ q ^ K := pow_nonneg hq K
      have hnum : q * (1 - q ^ K) ≤ q := by nlinarith [mul_nonneg hq hpow]
      exact mul_le_mul_of_nonneg_left
        (div_le_div_of_nonneg_right hnum hden.le) hd

/-- End-to-end finite readout bound for increments of one shared update-map orbit. -/
theorem shared_orbit_weighted_readout_bound
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (K : ℕ) (q : NNReal) (d : ℝ) (T : E → E) (x₀ : E)
    (w : Fin K → ℝ) (hq1 : (q : ℝ) < 1) (hd : 0 ≤ d)
    (hT : LipschitzWith q T)
    (hfirst : ‖T x₀ - x₀‖ ≤ (q : ℝ) * d)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w)
    (hsum : ∑ k, w k = K) :
    ‖∑ k, w k • ((T^[k.val + 1]) x₀ - (T^[k.val]) x₀)‖
      ≤ d * ((q : ℝ) * (1 - (q : ℝ) ^ K) / (1 - (q : ℝ))) := by
  apply weighted_increment_readout_bound K (q : ℝ) d w
    (fun k ↦ (T^[k.val + 1]) x₀ - (T^[k.val]) x₀)
    q.coe_nonneg hq1 hd hw0 hw hsum
  intro k
  simpa [dist_eq_norm] using
    shared_map_increment_decay q T hT x₀ d hd
      (by simpa [dist_eq_norm] using hfirst) k.val

/-- End-to-end uniform readout bound for increments of one shared update-map orbit. -/
theorem shared_orbit_weighted_readout_uniform
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (K : ℕ) (q : NNReal) (d : ℝ) (T : E → E) (x₀ : E)
    (w : Fin K → ℝ) (hq1 : (q : ℝ) < 1) (hd : 0 ≤ d)
    (hT : LipschitzWith q T)
    (hfirst : ‖T x₀ - x₀‖ ≤ (q : ℝ) * d)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w)
    (hsum : ∑ k, w k = K) :
    ‖∑ k, w k • ((T^[k.val + 1]) x₀ - (T^[k.val]) x₀)‖
      ≤ d * ((q : ℝ) / (1 - (q : ℝ))) := by
  apply weighted_increment_readout_uniform K (q : ℝ) d w
    (fun k ↦ (T^[k.val + 1]) x₀ - (T^[k.val]) x₀)
    q.coe_nonneg hq1 hd hw0 hw hsum
  intro k
  simpa [dist_eq_norm] using
    shared_map_increment_decay q T hT x₀ d hd
      (by simpa [dist_eq_norm] using hfirst) k.val

#print axioms state_recurrence_finite
#print axioms shifted_geom_sum
#print axioms state_recurrence_closed
#print axioms state_recurrence_uniform
#print axioms shared_map_increment_decay
#print axioms weighted_geometric_le_unweighted
#print axioms norm_weighted_sum_le
#print axioms weighted_increment_readout_bound
#print axioms weighted_increment_readout_uniform
#print axioms shared_orbit_weighted_readout_bound
#print axioms shared_orbit_weighted_readout_uniform

end NSI
