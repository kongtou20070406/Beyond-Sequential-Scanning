import NSI.Kernel
import NSI.Retention
import NSI.Readout

namespace NSI
open WithLp Function

abbrev GridState (P : Type*) (g : ℕ) := P × Fin g → PairState

/-- Each group has its own input-generated, cross-step fixed decay strength. -/
noncomputable def gridRetention {P : Type*} {g : ℕ}
    (α : P → ℝ) (h : GridState P g) : GridState P g :=
  fun pc => ofLp (radialRetention (α pc.1) (toLp 2 (fun c => h (pc.1,c)))) pc.2

theorem slice_norm_le {P : Type*} [Fintype P] {g : ℕ}
    (h : GridState P g) (p : P) : ‖fun c => h (p,c)‖ ≤ ‖h‖ := by
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg h)).2
  intro c
  exact norm_le_pi_norm h (p,c)

@[simp] theorem gridRetention_zero {P : Type*} {g : ℕ} (α : P → ℝ) :
    gridRetention α (0 : GridState P g) = 0 := by
  funext pc
  simp [gridRetention, radialRetention]

theorem gridRetention_norm_le {P : Type*} [Fintype P] {g : ℕ}
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (h : GridState P g) :
    ‖gridRetention α h‖ ≤ ‖h‖ := by
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg h)).2
  intro pc
  have hm := radialRetention_blockSup_magnitude_le (α pc.1) (hα pc.1)
    (toLp 2 (fun c => h (pc.1,c)))
  exact (norm_le_pi_norm _ pc.2).trans (hm.trans (slice_norm_le h pc.1))

noncomputable def gridUpdate {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (d h : GridState P g) : GridState P g :=
  neighborhoodAggregate a b neighbor (d + gridRetention α h)

theorem gridUpdate_magnitude {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d h : GridState P g) :
    ‖gridUpdate a b neighbor α d h‖ ≤ q * (‖d‖ + ‖h‖) := by
  calc
    _ ≤ q * ‖d + gridRetention α h‖ :=
      neighborhoodAggregate_norm_bound a b neighbor q q.coe_nonneg hgain _
    _ ≤ q * (‖d‖ + ‖gridRetention α h‖) :=
      mul_le_mul_of_nonneg_left (norm_add_le _ _) q.coe_nonneg
    _ ≤ q * (‖d‖ + ‖h‖) := by
      exact mul_le_mul_of_nonneg_left
        (add_le_add (le_refl ‖d‖) (gridRetention_norm_le α hα h)) q.coe_nonneg

theorem gridUpdate_first_bound {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d : GridState P g) :
    ‖gridUpdate a b neighbor α d 0‖ ≤ q * ‖d‖ := by
  simpa using gridUpdate_magnitude a b neighbor α hα q hgain d 0

/-- The concrete NSI state bound holds for every group size. -/
theorem gridState_uniform_bound {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (q : NNReal) (hq : (q : ℝ) < 1)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d : GridState P g) (K : ℕ) :
    ‖((gridUpdate a b neighbor α d)^[K]) 0‖ ≤ ‖d‖ * (q / (1-(q : ℝ))) := by
  apply state_recurrence_uniform (q : ℝ) ‖d‖
    (fun k => ‖((gridUpdate a b neighbor α d)^[k]) 0‖)
    q.coe_nonneg hq (norm_nonneg d)
  · simp
  · intro k
    simpa only [iterate_succ_apply'] using
      gridUpdate_magnitude a b neighbor α hα q hgain d
        (((gridUpdate a b neighbor α d)^[k]) 0)

#print axioms gridRetention_norm_le
#print axioms gridState_uniform_bound

theorem gridState_finite_bound {P : Type*} [Fintype P] {g : ℕ}
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (q : NNReal) (hq : (q : ℝ) < 1)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d h₀ : GridState P g) (K : ℕ) :
    ‖((gridUpdate a b neighbor α d)^[K]) h₀‖ ≤
      (q : ℝ)^K*‖h₀‖ + ‖d‖*((q : ℝ)*(1-(q : ℝ)^K)/(1-(q : ℝ))) := by
  let s := fun k => ‖((gridUpdate a b neighbor α d)^[k]) h₀‖
  have hh := state_recurrence_closed (q : ℝ) ‖d‖
    (fun k => s k - (q : ℝ)^k*‖h₀‖) q.coe_nonneg hq (norm_nonneg d)
    (by simp [s])
    (by
      intro k
      have hm := gridUpdate_magnitude a b neighbor α hα q hgain d
        (((gridUpdate a b neighbor α d)^[k]) h₀)
      have hm' : s (k+1) ≤ (q : ℝ)*(‖d‖+s k) := by
        simpa only [s, iterate_succ_apply'] using hm
      rw [pow_succ]
      nlinarith) K
  dsimp [s] at hh
  linarith

#print axioms gridState_finite_bound
end NSI
