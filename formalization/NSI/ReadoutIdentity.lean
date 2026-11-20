import Mathlib

namespace NSI
open Finset

/-- Expanding weighted state differences and collecting each intermediate state. -/
theorem weighted_difference_identity
    {E : Type*} [AddCommGroup E] [Module ℝ E]
    (w : ℕ → ℝ) (h : ℕ → E) (K : ℕ) :
    (∑ i ∈ range (K+1), w i • (h (i+1) - h i)) =
      w K • h (K+1) - w 0 • h 0 +
        ∑ i ∈ range K, (w i - w (i+1)) • h (i+1) := by
  induction K with
  | zero => simp [smul_sub]
  | succ K ih =>
      rw [sum_range_succ, ih, sum_range_succ]
      simp only [smul_sub, sub_smul]
      abel

/-- Equal weights recover the final state when the initial state is zero. -/
theorem equal_weight_readout
    {E : Type*} [AddCommGroup E] [Module ℝ E]
    (h : ℕ → E) (hzero : h 0 = 0) (K : ℕ) :
    (∑ i ∈ range K, (1 : ℝ) • (h (i+1) - h i)) = h K := by
  simp only [one_smul]
  rw [sum_range_sub, hzero, sub_zero]

#print axioms weighted_difference_identity
#print axioms equal_weight_readout
end NSI
