import NSI.Main

namespace NSI
open Finset

/-- The paper's K-scaled normalized exponential weights. -/
noncomputable def expReadoutWeight {K : ℕ} (score : Fin K → ℝ) (i : Fin K) : ℝ :=
  (K : ℝ) * Real.exp (score i) / ∑ j, Real.exp (score j)

theorem expReadoutWeight_nonneg {K : ℕ} (score : Fin K → ℝ) (i : Fin K) :
    0 ≤ expReadoutWeight score i := by
  unfold expReadoutWeight
  apply div_nonneg (mul_nonneg (Nat.cast_nonneg K) (Real.exp_pos _).le)
  exact sum_nonneg fun j _ => (Real.exp_pos (score j)).le

theorem expReadoutWeight_sum {K : ℕ} (score : Fin K → ℝ) :
    ∑ i, expReadoutWeight score i = K := by
  by_cases hK : K = 0
  · subst K
    simp
  · letI : NeZero K := ⟨hK⟩
    have hden : 0 < ∑ j, Real.exp (score j) :=
      sum_pos (fun j _ => Real.exp_pos (score j)) univ_nonempty
    simp only [expReadoutWeight]
    rw [← sum_div, ← mul_sum]
    exact mul_div_cancel_right₀ (K : ℝ) hden.ne'

theorem expReadoutWeight_monotone {K : ℕ} (score : Fin K → ℝ)
    (hs : Monotone score) : Monotone (expReadoutWeight score) := by
  intro i j hij
  unfold expReadoutWeight
  apply div_le_div_of_nonneg_right _ (sum_nonneg fun t _ => (Real.exp_pos (score t)).le)
  exact mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (hs hij)) (Nat.cast_nonneg K)

noncomputable def depthCoordinate {K : ℕ} (i : Fin K) : ℝ :=
  (i.val : ℝ) / ((K - 1 : ℕ) : ℝ)

theorem depthCoordinate_nonneg {K : ℕ} (i : Fin K) : 0 ≤ depthCoordinate i := by
  exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

theorem depthCoordinate_monotone {K : ℕ} : Monotone (@depthCoordinate K) := by
  intro i j hij
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg (K-1))
  exact_mod_cast hij

noncomputable def depthReadoutScore {K : ℕ} (θ₁ θ₂ : ℝ) (i : Fin K) : ℝ :=
  θ₁ ^ 2 * depthCoordinate i + θ₂ ^ 2 * (depthCoordinate i) ^ 2

theorem depthReadoutScore_monotone {K : ℕ} (θ₁ θ₂ : ℝ) :
    Monotone (@depthReadoutScore K θ₁ θ₂) := by
  intro i j hij
  have ht := depthCoordinate_monotone hij
  have hi := depthCoordinate_nonneg i
  have hj := depthCoordinate_nonneg j
  have hs : (depthCoordinate i)^2 ≤ (depthCoordinate j)^2 := by nlinarith
  exact add_le_add (mul_le_mul_of_nonneg_left ht (sq_nonneg θ₁))
    (mul_le_mul_of_nonneg_left hs (sq_nonneg θ₂))

theorem paperReadoutWeights_contract {K : ℕ} (θ₁ θ₂ : ℝ) :
    (∀ i : Fin K, 0 ≤ expReadoutWeight (depthReadoutScore θ₁ θ₂) i) ∧
    Monotone (expReadoutWeight (@depthReadoutScore K θ₁ θ₂)) ∧
    (∑ i : Fin K, expReadoutWeight (depthReadoutScore θ₁ θ₂) i) = K := by
  exact ⟨expReadoutWeight_nonneg _,
    expReadoutWeight_monotone _ (depthReadoutScore_monotone θ₁ θ₂),
    expReadoutWeight_sum _⟩

theorem paperReadoutWeights_singleton (θ₁ θ₂ : ℝ) :
    expReadoutWeight (@depthReadoutScore 1 θ₁ θ₂) 0 = 1 := by
  have hs := expReadoutWeight_sum (@depthReadoutScore 1 θ₁ θ₂)
  simpa using hs

theorem grid_readout_paper_weights
    {P : Type*} [Fintype P] {g : ℕ} [NeZero g]
    (a b : (P × Fin g) → Fin 9 → ℝ)
    (neighbor : (P × Fin g) → Fin 9 → Option (P × Fin g))
    (α : P → ℝ) (hα : ∀ p, 0 ≤ α p) (hg : g ≤ 4) (q : NNReal)
    (hgain : ∀ p, (∑ δ, Real.sqrt ((a p δ)^2+(b p δ)^2)) ≤ q)
    (d : GridState P g) (K : ℕ) (θ₁ θ₂ : ℝ) (hq : (q : ℝ)<1) :
    ‖∑ k : Fin K, expReadoutWeight (depthReadoutScore θ₁ θ₂) k •
       (((gridUpdate a b neighbor α d)^[k.val+1]) 0 -
        ((gridUpdate a b neighbor α d)^[k.val]) 0)‖ ≤
      ‖d‖ * ((q : ℝ)/(1-(q : ℝ))) := by
  apply grid_readout_uniform a b neighbor α hα hg q hgain d K
    (expReadoutWeight (depthReadoutScore θ₁ θ₂)) hq
  · exact expReadoutWeight_nonneg _
  · exact expReadoutWeight_monotone _ (depthReadoutScore_monotone θ₁ θ₂)
  · exact expReadoutWeight_sum _

#print axioms paperReadoutWeights_contract
#print axioms grid_readout_paper_weights
end NSI
