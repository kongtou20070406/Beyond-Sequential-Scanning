import NSI.Reflection
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv

namespace NSI

set_option maxHeartbeats 1000000
set_option synthInstance.maxHeartbeats 200000

open WithLp
open scoped RealInnerProductSpace

noncomputable def radialRetention {g : ℕ} (alpha : ℝ) (x : Blocks g) : Blocks g :=
  (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ • x

@[simp] theorem radialRetention_zero {g : ℕ} (alpha : ℝ) :
    radialRetention alpha (0 : Blocks g) = 0 := by
  simp [radialRetention]

theorem radialRetention_blockSup_magnitude_le {g : ℕ} (alpha : ℝ)
    (halpha : 0 ≤ alpha) (x : Blocks g) :
    ‖ofLp (radialRetention alpha x)‖ ≤ ‖ofLp x‖ := by
  have hbase : (1 : ℝ) ≤ 1 + alpha * ‖x‖ ^ 2 := by
    nlinarith [mul_nonneg halpha (sq_nonneg ‖x‖)]
  have hsqrt : 1 ≤ Real.sqrt (1 + alpha * ‖x‖ ^ 2) := by
    calc
      (1 : ℝ) = Real.sqrt 1 := by norm_num
      _ ≤ Real.sqrt (1 + alpha * ‖x‖ ^ 2) := Real.sqrt_le_sqrt hbase
  have hinv : 0 ≤ (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ :=
    inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hinv_le : (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ≤ 1 := by
    simpa using (inv_le_one₀ (by positivity : 0 < Real.sqrt (1 + alpha * ‖x‖ ^ 2))).2 hsqrt
  change ‖(Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ • ofLp x‖ ≤ _
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hinv]
  exact mul_le_of_le_one_left (norm_nonneg _) hinv_le

noncomputable def radialDeriv {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] (alpha : ℝ) (x : E) : E →L[ℝ] E :=
  (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ • ContinuousLinearMap.id ℝ E -
    ContinuousLinearMap.smulRight (M₁ := E) (M₂ := E)
      (R := ℝ) (S := ℝ) (innerSL ℝ x)
      ((alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 3) • x)

theorem hasFDerivAt_radial_generic {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] (alpha : ℝ) (halpha : 0 ≤ alpha) (x : E) :
    HasFDerivAt (fun y : E ↦ (Real.sqrt (1 + alpha * ‖y‖ ^ 2))⁻¹ • y)
      (radialDeriv alpha x) x := by
  let q : E → ℝ := fun y ↦ 1 + alpha * ‖y‖ ^ 2
  let t : ℝ := (Real.sqrt (q x))⁻¹
  have hq : 0 < q x := by dsimp [q]; positivity
  have hqderiv : HasFDerivAt q
      (alpha • (2 • innerSL ℝ x)) x := by
    simpa [q] using
      ((hasStrictFDerivAt_norm_sq x).hasFDerivAt.const_mul alpha).const_add 1
  have hsqrt : HasFDerivAt (fun y : E ↦ Real.sqrt (q y))
      ((2 * Real.sqrt (q x))⁻¹ • (alpha • (2 • innerSL ℝ x))) x := by
    simpa [one_div] using hqderiv.sqrt (ne_of_gt hq)
  have hinv := (hasFDerivAt_inv (Real.sqrt_ne_zero'.mpr hq)).comp x hsqrt
  have hsmul := hinv.smul (hasFDerivAt_id x)
  convert hsmul using 1
  · rfl
  · ext v
    simp only [radialDeriv, q, Function.comp_apply, id_eq,
      add_apply, sub_apply,
      smul_apply, ContinuousLinearMap.id_apply,
      ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.toSpanSingleton_apply]
    module

noncomputable abbrev radialRetentionDeriv {g : ℕ} (alpha : ℝ) (x : Blocks g) :=
  radialDeriv alpha x

theorem hasFDerivAt_radialRetention {g : ℕ} (alpha : ℝ)
    (halpha : 0 ≤ alpha) (x : Blocks g) :
    HasFDerivAt (radialRetention alpha) (radialRetentionDeriv alpha x) x := by
  exact hasFDerivAt_radial_generic alpha halpha x

theorem radialRetentionDeriv_blockSup_le {g : ℕ} [NeZero g]
    (alpha : ℝ) (halpha : 0 ≤ alpha) (hg4 : g ≤ 4) (x v : Blocks g) :
    ‖ofLp (radialRetentionDeriv alpha x v)‖ ≤ ‖ofLp v‖ := by
  by_cases hx : x = 0
  · subst x
    simp [radialRetentionDeriv, radialDeriv]
  · let r : ℝ := ‖x‖
    let d : ℝ := 1 + alpha * r ^ 2
    let u : Blocks g := r⁻¹ • x
    let s : ℝ := alpha * r ^ 2 / d
    have hr : 0 < r := by simp [r, norm_pos_iff, hx]
    have hd : 0 < d := by dsimp [d]; positivity
    have hu : ‖u‖ = 1 := by
      dsimp [u]
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hr)]
      simpa [r] using inv_mul_cancel₀ (norm_ne_zero_iff.mpr hx)
    have hs0 : 0 ≤ s := by dsimp [s]; positivity
    have hs1 : s ≤ 1 := by
      dsimp [s, d]
      apply (div_le_one hd).2
      linarith
    have hroot : Real.sqrt (1 - s) = (Real.sqrt d)⁻¹ := by
      have hone : 1 - s = d⁻¹ := by
        dsimp [s, d]
        field_simp
        ring
      rw [hone, Real.sqrt_inv]
    have hmix := jacobianMix_blockSup_le u v hu s hs0 hs1 hg4
    have heq : radialRetentionDeriv alpha x v =
        Real.sqrt (1 - s) •
          ((1 - s / 2) • v + (s / 2) • householder u v) := by
      rw [householder_eq u v hu]
      simp only [radialRetentionDeriv, radialDeriv, sub_apply,
        smul_apply, ContinuousLinearMap.id_apply,
        ContinuousLinearMap.smulRight_apply, hroot]
      dsimp [u, s, d, r]
      have hsqrt_ne : Real.sqrt (1 + alpha * ‖x‖ ^ 2) ≠ 0 := by positivity
      have hnorm_ne : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
      have hsqrt_sq : Real.sqrt (1 + alpha * ‖x‖ ^ 2) ^ 2 =
          1 + alpha * ‖x‖ ^ 2 := Real.sq_sqrt (by positivity)
      have hden_ne : 2 + alpha * ‖x‖ ^ 2 * 2 ≠ 0 := by positivity
      have ht_sq : (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 2 =
          (1 + alpha * ‖x‖ ^ 2)⁻¹ := by
        rw [inv_pow, hsqrt_sq]
      have hr_cancel : ‖x‖ ^ 2 * ‖x‖⁻¹ ^ 2 = 1 := by
        field_simp [hnorm_ne]
      have hscalar :
          -(⟪x, v⟫ * alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 3) =
            -(⟪x, v⟫ * alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ *
              ‖x‖ ^ 2 * (1 + alpha * ‖x‖ ^ 2)⁻¹ * ‖x‖⁻¹ ^ 2) := by
        rw [show (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 3 =
          (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ *
            (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 2 by ring, ht_sq]
        field_simp [hnorm_ne]
      have hscalar_pos :
          ⟪x, v⟫ * alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 3 =
            ⟪x, v⟫ * alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ *
              ‖x‖ ^ 2 * (1 + alpha * ‖x‖ ^ 2)⁻¹ * ‖x‖⁻¹ ^ 2 := by
        exact neg_injective hscalar
      have hcoefx :
          ⟪x, v⟫ * (alpha * (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ ^ 3) =
            (Real.sqrt (1 + alpha * ‖x‖ ^ 2))⁻¹ *
              (alpha * ‖x‖ ^ 2 / (1 + alpha * ‖x‖ ^ 2) / 2 *
                (2 * (‖x‖⁻¹ * ⟪x, v⟫ * ‖x‖⁻¹))) := by
        rw [← mul_assoc, hscalar_pos]
        field_simp [hnorm_ne]
      simp_rw [real_inner_smul_left]
      simp only [smul_add, smul_sub, smul_smul]
      rw [hcoefx]
      module
    rw [heq]
    exact hmix

#print axioms radialRetention_zero
#print axioms radialRetention_blockSup_magnitude_le
#print axioms hasFDerivAt_radialRetention
#print axioms radialRetentionDeriv_blockSup_le

end NSI
