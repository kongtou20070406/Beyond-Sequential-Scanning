import NSI.Grid
import NSI.Extensions

namespace NSI
open WithLp

theorem radialRetention_saturation {g : ℕ} (α : ℝ) (hα : 0 < α) (x : Blocks g) :
    ‖ofLp (radialRetention α x)‖ ≤ 1 / Real.sqrt α := by
  calc
    ‖ofLp (radialRetention α x)‖ ≤ ‖radialRetention α x‖ := by
      apply (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2
      intro i
      exact PiLp.norm_apply_le _ i
    _ = positiveRetention α ‖x‖ := by
      simp [radialRetention, positiveRetention, norm_smul,
        div_eq_mul_inv, mul_comm]
    _ ≤ 1 / Real.sqrt α := positiveRetention_le_inv_sqrt hα (norm_nonneg x)

theorem gridRetention_saturation {P : Type*} [Fintype P] {g : ℕ}
    (α : P → ℝ) (μ : ℝ) (hμ : 0 < μ) (hmin : ∀ p, μ ≤ α p)
    (h : GridState P g) :
    ‖gridRetention α h‖ ≤ 1 / Real.sqrt μ := by
  apply (pi_norm_le_iff_of_nonneg (by positivity)).2
  intro pc
  have ha : 0 < α pc.1 := hμ.trans_le (hmin pc.1)
  have hs := radialRetention_saturation (α pc.1) ha
    (toLp 2 (fun c => h (pc.1,c)))
  calc
    ‖gridRetention α h pc‖ ≤
        ‖ofLp (radialRetention (α pc.1) (toLp 2 (fun c => h (pc.1,c))))‖ :=
      norm_le_pi_norm _ pc.2
    _ ≤ 1 / Real.sqrt (α pc.1) := hs
    _ ≤ 1 / Real.sqrt μ :=
      one_div_le_one_div_of_le (Real.sqrt_pos.mpr hμ) (Real.sqrt_le_sqrt (hmin pc.1))

/-- The actual NSI old-state contribution is uniformly bounded under positive decay. -/
theorem gridUpdate_old_contribution_bound {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (μ : ℝ) (hμ : 0 < μ) (hmin : ∀ p, μ ≤ α p)
    (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d h : GridState P g) :
    ‖gridUpdate a b neighbor α d h - gridUpdate a b neighbor α d 0‖ ≤
      q * (1 / Real.sqrt μ) := by
  have hL := neighborhoodAggregate_lipschitz a b neighbor q hgain
  calc
    _ ≤ q * ‖(d + gridRetention α h) - (d + gridRetention α 0)‖ := by
      simpa only [gridUpdate, dist_eq_norm] using
        hL.dist_le_mul (d + gridRetention α h) (d + gridRetention α 0)
    _ = q * ‖gridRetention α h‖ := by simp
    _ ≤ q * (1 / Real.sqrt μ) :=
      mul_le_mul_of_nonneg_left (gridRetention_saturation α μ hμ hmin h) q.coe_nonneg

theorem gridUpdate_saturation_bound {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (μ : ℝ) (hμ : 0 < μ) (hmin : ∀ p, μ ≤ α p)
    (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d h : GridState P g) :
    ‖gridUpdate a b neighbor α d h‖ ≤ q * (‖d‖ + 1 / Real.sqrt μ) := by
  calc
    _ ≤ q * ‖d + gridRetention α h‖ :=
      neighborhoodAggregate_norm_bound a b neighbor q q.coe_nonneg hgain _
    _ ≤ q * (‖d‖ + ‖gridRetention α h‖) :=
      mul_le_mul_of_nonneg_left (norm_add_le _ _) q.coe_nonneg
    _ ≤ q * (‖d‖ + 1 / Real.sqrt μ) :=
      mul_le_mul_of_nonneg_left
        (add_le_add (le_refl _) (gridRetention_saturation α μ hμ hmin h)) q.coe_nonneg

#print axioms radialRetention_saturation
#print axioms gridRetention_saturation
#print axioms gridUpdate_old_contribution_bound
#print axioms gridUpdate_saturation_bound
end NSI
