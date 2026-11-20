import Mathlib

namespace NSI

open scoped ENNReal
open WithLp

/-- Exact finite-dimensional comparison used for the NSI block maximum norm:
the L2 norm of the flattened blocks is at most sqrt(number of blocks) times
the ordinary Pi (supremum) norm. -/
theorem piLp_two_norm_le_sqrt_card_mul_pi_norm
    {ι : Type*} [Fintype ι] [Nonempty ι]
    {E : ι → Type*} [∀ i, SeminormedAddCommGroup (E i)]
    (v : ∀ i, E i) :
    ‖toLp 2 v‖ ≤ Real.sqrt (Fintype.card ι) * ‖v‖ := by
  rw [PiLp.norm_eq_of_L2]
  have hsum :
      (∑ i : ι, ‖v i‖ ^ 2) ≤
        (Fintype.card ι : ℝ) * ‖v‖ ^ 2 := by
    calc
      (∑ i : ι, ‖v i‖ ^ 2) ≤ ∑ _i : ι, ‖v‖ ^ 2 := by
        apply Finset.sum_le_sum
        intro i hi
        nlinarith [norm_nonneg (v i), norm_nonneg v, norm_le_pi_norm v i]
      _ = (Fintype.card ι : ℝ) * ‖v‖ ^ 2 := by simp
  calc
    Real.sqrt (∑ i : ι, ‖v i‖ ^ 2)
        ≤ Real.sqrt ((Fintype.card ι : ℝ) * ‖v‖ ^ 2) :=
      Real.sqrt_le_sqrt hsum
    _ = Real.sqrt (Fintype.card ι) * ‖v‖ := by
      rw [Real.sqrt_mul (Nat.cast_nonneg _), Real.sqrt_sq_eq_abs, abs_norm]

#print axioms piLp_two_norm_le_sqrt_card_mul_pi_norm

end NSI
