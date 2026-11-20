import NSI.Saturation

namespace NSI

theorem positiveRetention_hasDerivAt (α r : ℝ) (hα : 0 ≤ α) :
    HasDerivAt (positiveRetention α) ((Real.sqrt (1+α*r^2))⁻¹^3) r := by
  have hp : 0 < 1+α*r^2 := by positivity
  have hs := (((hasDerivAt_id r).pow 2).const_mul α).const_add 1
  have hd := (hasDerivAt_id r).div (hs.sqrt (ne_of_gt hp))
    (Real.sqrt_ne_zero'.mpr hp)
  simp only [id_eq, Pi.pow_apply, Nat.cast_ofNat, Nat.reduceSub, pow_one,
    mul_one, one_mul] at hd
  change HasDerivAt (positiveRetention α)
    ((Real.sqrt (1+α*r^2)-r*(α*(2*r)/(2*Real.sqrt (1+α*r^2)))) /
       Real.sqrt (1+α*r^2)^2) r at hd
  convert hd using 1
  have hsq := Real.sq_sqrt hp.le
  have hn := Real.sqrt_ne_zero'.mpr hp
  field_simp
  nlinarith

theorem positiveRetention_monotone (α : ℝ) (hα : 0 ≤ α) :
    Monotone (positiveRetention α) := by
  apply monotone_of_deriv_nonneg
  · intro r; exact (positiveRetention_hasDerivAt α r hα).differentiableAt
  · intro r; rw [(positiveRetention_hasDerivAt α r hα).deriv]; positivity

theorem positiveRetention_lipschitz (α : ℝ) (hα : 0 ≤ α) :
    LipschitzWith 1 (positiveRetention α) := by
  apply lipschitzWith_of_nnnorm_deriv_le
  · intro r; exact (positiveRetention_hasDerivAt α r hα).differentiableAt
  · intro r
    rw [(positiveRetention_hasDerivAt α r hα).deriv]
    change ‖(Real.sqrt (1+α*r^2))⁻¹^3‖ ≤ (1 : ℝ)
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    have hs : 1 ≤ Real.sqrt (1+α*r^2) := by
      apply Real.le_sqrt_of_sq_le; nlinarith [mul_nonneg hα (sq_nonneg r)]
    have hi : 0 ≤ (Real.sqrt (1+α*r^2))⁻¹ := by positivity
    have hb : (Real.sqrt (1+α*r^2))⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hs
    exact pow_le_one₀ hi hb

noncomputable def scalarEnvelope (q d α r : ℝ) := q * (d + positiveRetention α r)

theorem scalarEnvelope_lipschitz (q : NNReal) (d α : ℝ) (hα : 0 ≤ α) :
    LipschitzWith q (scalarEnvelope q d α) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  have h := (positiveRetention_lipschitz α hα).dist_le_mul x y
  simp only [NNReal.coe_one, one_mul, dist_eq_norm] at h
  simpa [scalarEnvelope, dist_eq_norm, ← mul_sub, norm_mul,
    Real.norm_eq_abs, abs_of_nonneg q.coe_nonneg] using
    mul_le_mul_of_nonneg_left h q.coe_nonneg

theorem scalarEnvelope_unique_fixedPoint (q : NNReal) (hq : (q : ℝ) < 1)
    (d α : ℝ) (hα : 0 ≤ α) : ∃! B : ℝ, scalarEnvelope q d α B = B := by
  have hc : ContractingWith q (scalarEnvelope q d α) :=
    ⟨hq, scalarEnvelope_lipschitz q d α hα⟩
  exact ⟨_, hc.fixedPoint_isFixedPt, fun y hy => hc.fixedPoint_unique hy⟩

#print axioms positiveRetention_hasDerivAt
#print axioms positiveRetention_lipschitz
#print axioms scalarEnvelope_unique_fixedPoint

theorem scalarEnvelope_fixedPoint_nonneg (q : NNReal) (hq : (q : ℝ) < 1)
    (d α B : ℝ) (hd : 0 ≤ d) (hα : 0 ≤ α)
    (hfix : scalarEnvelope q d α B = B) : 0 ≤ B := by
  by_contra hn
  have hB : B < 0 := lt_of_not_ge hn
  have hm := positiveRetention_le_self hα (neg_nonneg.mpr hB.le)
  have hodd : positiveRetention α (-B) = -positiveRetention α B := by
    simp [positiveRetention, neg_div]
  rw [hodd] at hm
  have hlow : B ≤ positiveRetention α B := by linarith
  have hprod := mul_le_mul_of_nonneg_left hlow q.coe_nonneg
  have hqd := mul_nonneg q.coe_nonneg hd
  have hstrict : B < (q : ℝ)*B := by nlinarith
  dsimp [scalarEnvelope] at hfix
  nlinarith

theorem scalarEnvelope_positive_fixedPoint (q : NNReal) (hq0 : 0 < (q : ℝ))
    (hq : (q : ℝ) < 1) (d α : ℝ) (hd : 0 < d) (hα : 0 < α) :
    ∃ B : ℝ, scalarEnvelope q d α B = B ∧
      0 < B ∧ B < q*d/(1-(q : ℝ)) ∧
      ∀ C, scalarEnvelope q d α C = C → C = B := by
  obtain ⟨B,hB,hu⟩ := scalarEnvelope_unique_fixedPoint q hq d α hα.le
  have hn := scalarEnvelope_fixedPoint_nonneg q hq d α B hd.le hα.le hB
  have hb := positiveRetention_fixedPoint_lt_geometric hq0 hq hd hα hn hB.symm
  exact ⟨B,hB,hb.1,hb.2,hu⟩

open WithLp Function in
theorem gridRetention_scalar_envelope {P : Type*} [Fintype P] {g : ℕ}
    (α : P → ℝ) (μ : ℝ) (hμ : 0 ≤ μ) (hmin : ∀ p, μ ≤ α p)
    (h : GridState P g) :
    ‖gridRetention α h‖ ≤ positiveRetention μ ‖h‖ := by
  apply (pi_norm_le_iff_of_nonneg (by unfold positiveRetention; positivity)).2
  intro pc
  let x : Blocks g := toLp 2 (fun c => h (pc.1,c))
  have hc : ‖h pc‖ ≤ ‖x‖ := PiLp.norm_apply_le x pc.2
  have ha : 0 ≤ α pc.1 := hμ.trans (hmin pc.1)
  have hsq : μ*‖h pc‖^2 ≤ α pc.1*‖x‖^2 :=
    mul_le_mul (hmin pc.1) (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)|>.2 hc)
      (sq_nonneg _) ha
  have hden := Real.sqrt_le_sqrt (add_le_add (le_refl (1 : ℝ)) hsq)
  have hpos := positiveRetention_den_pos (α := μ) (r := ‖h pc‖) hμ
  calc
    ‖gridRetention α h pc‖ = ‖h pc‖ / Real.sqrt (1+α pc.1*‖x‖^2) := by
      simp [gridRetention, radialRetention, x, norm_smul,
        div_eq_mul_inv, mul_comm]
    _ ≤ positiveRetention μ ‖h pc‖ :=
      div_le_div_of_nonneg_left (norm_nonneg _) hpos hden
    _ ≤ positiveRetention μ ‖h‖ :=
      positiveRetention_monotone μ hμ (norm_le_pi_norm h pc)

#print axioms scalarEnvelope_positive_fixedPoint
#print axioms gridRetention_scalar_envelope

open Function in
theorem gridState_scalar_envelope {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (μ : ℝ) (hμ : 0 ≤ μ) (hmin : ∀ p, μ ≤ α p)
    (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d : GridState P g) (K : ℕ) :
    ‖((gridUpdate a b neighbor α d)^[K]) 0‖ ≤
      ((scalarEnvelope q ‖d‖ μ)^[K]) 0 := by
  induction K with
  | zero => simp
  | succ K ih =>
    rw [iterate_succ_apply', iterate_succ_apply']
    let h := ((gridUpdate a b neighbor α d)^[K]) 0
    change ‖gridUpdate a b neighbor α d h‖ ≤ _
    calc
      _ ≤ q * ‖d + gridRetention α h‖ :=
        neighborhoodAggregate_norm_bound a b neighbor q q.coe_nonneg hgain _
      _ ≤ q * (‖d‖ + ‖gridRetention α h‖) :=
        mul_le_mul_of_nonneg_left (norm_add_le _ _) q.coe_nonneg
      _ ≤ q * (‖d‖ + positiveRetention μ ‖h‖) :=
        mul_le_mul_of_nonneg_left
          (add_le_add (le_refl _) (gridRetention_scalar_envelope α μ hμ hmin h))
          q.coe_nonneg
      _ ≤ scalarEnvelope q ‖d‖ μ (((scalarEnvelope q ‖d‖ μ)^[K]) 0) :=
        mul_le_mul_of_nonneg_left
          (add_le_add (le_refl _) (positiveRetention_monotone μ hμ ih)) q.coe_nonneg

#print axioms gridState_scalar_envelope
end NSI
