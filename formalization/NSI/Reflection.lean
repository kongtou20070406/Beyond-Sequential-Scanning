import NSI.MixedNorm
import NSI.Radial
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection

namespace NSI

open WithLp

abbrev Blocks (g : ℕ) := PiLp 2 (fun _ : Fin g => EuclideanSpace ℝ (Fin 2))

noncomputable def householder {g : ℕ} (u : Blocks g) :
    Blocks g ≃ₗᵢ[ℝ] Blocks g :=
  (ℝ ∙ u)ᗮ.reflection

theorem householder_eq {g : ℕ} (u v : Blocks g) (hu : ‖u‖ = 1) :
    householder u v = v - (2 : ℝ) • ((inner ℝ u v) • u) := by
  rw [householder, Submodule.reflection_orthogonal_apply,
    Submodule.reflection_singleton_apply, hu]
  simp
  module

theorem householder_blockSup_le {g : ℕ} [NeZero g]
    (u v : Blocks g) (_hu : ‖u‖ = 1) :
    ‖ofLp (householder u v)‖ ≤ Real.sqrt g * ‖ofLp v‖ := by
  have hsup_l2 : ‖ofLp (householder u v)‖ ≤ ‖householder u v‖ := by
    rw [pi_norm_le_iff_of_nonneg (norm_nonneg _)]
    intro i
    simpa [dist_zero_right] using
      (PiLp.dist_apply_le (householder u v) 0 i)
  calc
    ‖ofLp (householder u v)‖ ≤ ‖householder u v‖ := hsup_l2
    _ = ‖v‖ := (householder u).norm_map v
    _ ≤ Real.sqrt (Fintype.card (Fin g)) * ‖ofLp v‖ :=
      piLp_two_norm_le_sqrt_card_mul_pi_norm (ofLp v)
    _ = Real.sqrt g * ‖ofLp v‖ := by simp

theorem jacobianMix_blockSup_le {g : ℕ} [NeZero g]
    (u v : Blocks g) (hu : ‖u‖ = 1) (s : ℝ)
    (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (hg4 : g ≤ 4) :
    ‖ofLp
        (Real.sqrt (1 - s) •
          ((1 - s / 2) • v + (s / 2) • householder u v))‖
      ≤ ‖ofLp v‖ := by
  have hs2 : 0 ≤ s / 2 := by positivity
  have hcoef : 0 ≤ 1 - s / 2 := by linarith
  have hroot : 0 ≤ Real.sqrt (1 - s) := Real.sqrt_nonneg _
  have hH := householder_blockSup_le u v hu
  have hfirst :
      ‖ofLp ((1 - s / 2) • v)‖ = (1 - s / 2) * ‖ofLp v‖ := by
    change ‖(1 - s / 2) • ofLp v‖ = _
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hcoef]
  have hsecond :
      ‖ofLp ((s / 2) • householder u v)‖ =
        (s / 2) * ‖ofLp (householder u v)‖ := by
    change ‖(s / 2) • ofLp (householder u v)‖ = _
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hs2]
  have hinside :
      ‖ofLp ((1 - s / 2) • v + (s / 2) • householder u v)‖
        ≤ (1 + ((Real.sqrt g - 1) / 2) * s) * ‖ofLp v‖ := by
    calc
      ‖ofLp ((1 - s / 2) • v + (s / 2) • householder u v)‖
          ≤ ‖ofLp ((1 - s / 2) • v)‖ +
              ‖ofLp ((s / 2) • householder u v)‖ := norm_add_le _ _
      _ = (1 - s / 2) * ‖ofLp v‖ +
              (s / 2) * ‖ofLp (householder u v)‖ := by
            rw [hfirst, hsecond]
      _ ≤ (1 - s / 2) * ‖ofLp v‖ +
              (s / 2) * (Real.sqrt g * ‖ofLp v‖) := by
            exact add_le_add_right (mul_le_mul_of_nonneg_left hH hs2) _
      _ = (1 + ((Real.sqrt g - 1) / 2) * s) * ‖ofLp v‖ := by ring
  calc
    ‖ofLp
        (Real.sqrt (1 - s) •
          ((1 - s / 2) • v + (s / 2) • householder u v))‖
        = Real.sqrt (1 - s) *
            ‖ofLp ((1 - s / 2) • v + (s / 2) • householder u v)‖ := by
          change
            ‖Real.sqrt (1 - s) •
                ofLp ((1 - s / 2) • v + (s / 2) • householder u v)‖ = _
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hroot]
    _ ≤ Real.sqrt (1 - s) *
          ((1 + ((Real.sqrt g - 1) / 2) * s) * ‖ofLp v‖) := by
          gcongr
    _ = (Real.sqrt (1 - s) *
          (1 + ((Real.sqrt g - 1) / 2) * s)) * ‖ofLp v‖ := by ring
    _ ≤ 1 * ‖ofLp v‖ := by
          gcongr
          apply grouped_radial_factor_le_one
          · exact_mod_cast (Nat.one_le_iff_ne_zero.mpr (NeZero.ne g))
          · exact_mod_cast hg4
          · exact hs0
          · exact hs1
    _ = ‖ofLp v‖ := one_mul _

#print axioms householder_eq
#print axioms householder_blockSup_le
#print axioms jacobianMix_blockSup_le

end NSI
