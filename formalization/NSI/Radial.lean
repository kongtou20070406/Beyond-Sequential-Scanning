import Mathlib

namespace NSI

/-- Scalar inequality used in the Jacobian estimate for grouped radial
retention.  In the NSI application, `a = (sqrt g - 1) / 2`; `g ≤ 4`
implies `a ≤ 1/2`. -/
theorem radial_factor_le_one
    (s a : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1 / 2) :
    Real.sqrt (1 - s) * (1 + a * s) ≤ 1 := by
  have hroot0 : 0 ≤ Real.sqrt (1 - s) := Real.sqrt_nonneg _
  have hbase0 : 0 ≤ 1 - s := sub_nonneg.mpr hs1
  have hroot_sq : (Real.sqrt (1 - s)) ^ 2 = 1 - s := by
    simpa using Real.sq_sqrt hbase0
  have hfac0 : 0 ≤ 1 + a * s := by positivity
  have hfac_le : 1 + a * s ≤ 1 + s / 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr ha1) hs0]
  have hprod_le :
      Real.sqrt (1 - s) * (1 + a * s)
        ≤ Real.sqrt (1 - s) * (1 + s / 2) :=
    mul_le_mul_of_nonneg_left hfac_le hroot0
  have hsq : (Real.sqrt (1 - s) * (1 + s / 2)) ^ 2 ≤ 1 := by
    rw [mul_pow, hroot_sq]
    nlinarith [sq_nonneg s, mul_nonneg hs0 (sub_nonneg.mpr hs1)]
  have hprod0 : 0 ≤ Real.sqrt (1 - s) * (1 + s / 2) := by positivity
  have hprod : Real.sqrt (1 - s) * (1 + s / 2) ≤ 1 := by
    nlinarith [sq_nonneg (Real.sqrt (1 - s) * (1 + s / 2) - 1)]
  exact hprod_le.trans hprod

/-- The paper's `g ≤ 4` specialization of the scalar Jacobian factor. -/
theorem grouped_radial_factor_le_one
    (g s : ℝ) (hg1 : 1 ≤ g) (hg4 : g ≤ 4)
    (hs0 : 0 ≤ s) (hs1 : s ≤ 1) :
    Real.sqrt (1 - s) *
        (1 + ((Real.sqrt g - 1) / 2) * s) ≤ 1 := by
  have hg0 : 0 ≤ g := le_trans (by norm_num) hg1
  have hsqrt1 : 1 ≤ Real.sqrt g := by
    rw [← Real.sqrt_one]
    exact Real.sqrt_le_sqrt hg1
  have hsqrt4 : Real.sqrt g ≤ 2 := by
    have := Real.sqrt_le_sqrt hg4
    norm_num at this ⊢
    exact this
  apply radial_factor_le_one s ((Real.sqrt g - 1) / 2) hs0 hs1
  · linarith
  · linarith

#print axioms radial_factor_le_one
#print axioms grouped_radial_factor_le_one

end NSI
