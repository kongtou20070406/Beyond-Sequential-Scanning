import NSI.Grid
import NSI.RetentionLipschitz

namespace NSI
open WithLp Function

/-- `P` combines batch, spatial position, and group index; `Fin g` indexes pairs in a group. -/
theorem gridRetention_lipschitz {P : Type*} [Fintype P] {g : ℕ} [NeZero g]
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (hg : g ≤ 4) :
    LipschitzWith 1 (gridRetention (g := g) α) := by
  apply LipschitzWith.of_dist_le_mul
  intro h v
  simp only [NNReal.coe_one, one_mul, dist_eq_norm]
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg (h-v))).2
  intro pc
  have hr := radialRetention_blockSup_nonexpansive (α pc.1) (hα pc.1) hg
    (toLp 2 (fun c => h (pc.1,c))) (toLp 2 (fun c => v (pc.1,c)))
  have hh := (norm_le_pi_norm _ pc.2).trans hr
  exact hh.trans (slice_norm_le (h-v) pc.1)

section Concrete
variable {P : Type*} [Fintype P] {g : ℕ} [NeZero g]
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (hg : g ≤ 4) (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d : GridState P g)
include hα hg hgain

theorem gridUpdate_lipschitz : LipschitzWith q (gridUpdate a b neighbor α d) := by
  apply LipschitzWith.of_dist_le_mul
  intro h v
  have hR := (gridRetention_lipschitz α hα hg).dist_le_mul h v
  have hL := (neighborhoodAggregate_lipschitz a b neighbor q hgain).dist_le_mul
    (d+gridRetention α h) (d+gridRetention α v)
  simp only [dist_add_left] at hL
  exact hL.trans (mul_le_mul_of_nonneg_left (by simpa using hR) q.coe_nonneg)

theorem grid_trajectories_contract (h v : GridState P g) (K : ℕ) :
    ‖((gridUpdate a b neighbor α d)^[K]) h -
       ((gridUpdate a b neighbor α d)^[K]) v‖ ≤ (q : ℝ)^K * ‖h-v‖ := by
  simpa only [dist_eq_norm, NNReal.coe_pow] using
    ((gridUpdate_lipschitz a b neighbor α hα hg q hgain d).iterate K).dist_le_mul h v

theorem grid_unique_fixedPoint (hq : (q : ℝ)<1) :
    ∃! h : GridState P g, gridUpdate a b neighbor α d h = h := by
  have hc : ContractingWith q (gridUpdate a b neighbor α d) :=
    ⟨hq,gridUpdate_lipschitz a b neighbor α hα hg q hgain d⟩
  exact ⟨_,hc.fixedPoint_isFixedPt,fun y hy => hc.fixedPoint_unique hy⟩

theorem grid_readout_finite (K : ℕ) (w : Fin K → ℝ) (hq : (q : ℝ)<1)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w) (hsum : ∑ k, w k = K) :
    ‖∑ k, w k • (((gridUpdate a b neighbor α d)^[k.val+1]) 0 -
                  ((gridUpdate a b neighbor α d)^[k.val]) 0)‖ ≤
      ‖d‖ * ((q : ℝ)*(1-(q : ℝ)^K)/(1-(q : ℝ))) := by
  apply shared_orbit_weighted_readout_bound K q ‖d‖
    (gridUpdate a b neighbor α d) 0 w hq (norm_nonneg d)
    (gridUpdate_lipschitz a b neighbor α hα hg q hgain d)
    _ hw0 hw hsum
  simpa using gridUpdate_first_bound a b neighbor α hα q hgain d

theorem grid_readout_uniform (K : ℕ) (w : Fin K → ℝ) (hq : (q : ℝ)<1)
    (hw0 : ∀ k, 0 ≤ w k) (hw : Monotone w) (hsum : ∑ k, w k = K) :
    ‖∑ k, w k • (((gridUpdate a b neighbor α d)^[k.val+1]) 0 -
                  ((gridUpdate a b neighbor α d)^[k.val]) 0)‖ ≤
      ‖d‖ * ((q : ℝ)/(1-(q : ℝ))) := by
  apply shared_orbit_weighted_readout_uniform K q ‖d‖
    (gridUpdate a b neighbor α d) 0 w hq (norm_nonneg d)
    (gridUpdate_lipschitz a b neighbor α hα hg q hgain d)
    _ hw0 hw hsum
  simpa using gridUpdate_first_bound a b neighbor α hα q hgain d
end Concrete

theorem normalized_gridUpdate_lipschitz {P : Type*} [Fintype P] {g : ℕ} [NeZero g]
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (hg : g ≤ 4)
    (ρ : NNReal) (hρ : 0 < (ρ : ℝ)) (d : GridState P g) :
    LipschitzWith ρ (gridUpdate (normalizeKernel a b ρ) (normalizeKernel b a ρ)
      neighbor α d) :=
  gridUpdate_lipschitz _ _ neighbor α hα hg ρ
    (fun p => (normalizedKernel_gain a b ρ hρ p).le) d

#print axioms gridRetention_lipschitz
#print axioms gridUpdate_lipschitz
#print axioms grid_trajectories_contract
#print axioms grid_unique_fixedPoint
#print axioms grid_readout_finite
#print axioms grid_readout_uniform
#print axioms normalized_gridUpdate_lipschitz
end NSI
