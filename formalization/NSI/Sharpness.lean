import NSI.Retention

namespace NSI

open Finset
open WithLp

/-! An explicit five-block scalar witness.  Each scalar is the first coordinate
of a two-dimensional block; the second coordinate is zero. -/

noncomputable def sharpX (i : Fin 5) : ℝ := if i = 0 then 1 / 5 else 2 / 15
noncomputable def sharpV (i : Fin 5) : ℝ := if i = 0 then 1 else -1

def scalarBlockSup (v : Fin 5 → ℝ) : ℝ := ‖v‖

noncomputable def sharpNormSq : ℝ := ∑ i : Fin 5, sharpX i ^ 2
noncomputable def sharpDot : ℝ := ∑ i : Fin 5, sharpX i * sharpV i

noncomputable def sharpJacobian (i : Fin 5) : ℝ :=
  let t := (Real.sqrt (1 + sharpNormSq))⁻¹
  t * sharpV i - t ^ 3 * sharpDot * sharpX i

noncomputable def embedScalar5 (a : Fin 5 → ℝ) : Blocks 5 :=
  toLp 2 (fun i ↦ toLp 2 (fun j : Fin 2 ↦ if j = 0 then a i else 0))

noncomputable def actualX5 : Blocks 5 := embedScalar5 sharpX
noncomputable def actualV5 : Blocks 5 := embedScalar5 sharpV

lemma sharpNormSq_eq : sharpNormSq = 1 / 9 := by
  norm_num [sharpNormSq, sharpX, Fin.sum_univ_succ]

lemma sharpDot_eq : sharpDot = -1 / 3 := by
  norm_num [sharpDot, sharpX, sharpV, Fin.sum_univ_succ]

lemma actualX5_norm_sq : ‖actualX5‖ ^ 2 = 1 / 9 := by
  have h0 : sharpX (0 : Fin 5) = 1 / 5 := by norm_num [sharpX]
  have h1 : sharpX (1 : Fin 5) = 2 / 15 := by norm_num [sharpX]
  have h2 : sharpX (2 : Fin 5) = 2 / 15 := by
    simp [sharpX, show (2 : Fin 5) ≠ 0 by decide]
  have h3 : sharpX (3 : Fin 5) = 2 / 15 := by
    simp [sharpX, show (3 : Fin 5) ≠ 0 by decide]
  have h4 : sharpX (4 : Fin 5) = 2 / 15 := by
    simp [sharpX, show (4 : Fin 5) ≠ 0 by decide]
  rw [show ‖actualX5‖ ^ 2 = ∑ i : Fin 5, sharpX i ^ 2 by
    simp [actualX5, embedScalar5, PiLp.norm_eq_of_L2, Fin.sum_univ_succ]
    rw [h0, h1, h2, h3, h4]
    norm_num]
  exact sharpNormSq_eq

lemma actual_inner_eq : inner ℝ actualX5 actualV5 = -1 / 3 := by
  rw [show inner ℝ actualX5 actualV5 = ∑ i : Fin 5, sharpX i * sharpV i by
    simp [actualX5, actualV5, embedScalar5, PiLp.inner_apply, Fin.sum_univ_succ]
    ring]
  exact sharpDot_eq

lemma sharpV_blockSup_eq : scalarBlockSup sharpV = 1 := by
  unfold scalarBlockSup
  apply le_antisymm
  · rw [pi_norm_le_iff_of_nonneg (by norm_num)]
    intro i
    fin_cases i <;> norm_num [sharpV]
  · simpa [scalarBlockSup, sharpV] using norm_le_pi_norm sharpV (0 : Fin 5)

lemma sharpJacobian_first_eq :
    sharpJacobian 0 = 159 / (50 * Real.sqrt 10) := by
  rw [sharpJacobian, sharpNormSq_eq, sharpDot_eq]
  have hs10 : Real.sqrt 10 ≠ 0 := by positivity
  have hs10sq : Real.sqrt 10 ^ 2 = 10 := Real.sq_sqrt (by norm_num)
  have hs109 : Real.sqrt (10 / 9) = Real.sqrt 10 / 3 := by
    rw [Real.sqrt_div (by norm_num) 9]
    norm_num
  rw [show 1 + (1 / 9 : ℝ) = 10 / 9 by norm_num, hs109]
  simp only [sharpV, sharpX, if_pos]
  field_simp
  nlinarith

lemma sharpJacobian_first_gt_one : 1 < sharpJacobian 0 := by
  rw [sharpJacobian_first_eq]
  have hs10pos : 0 < Real.sqrt 10 := Real.sqrt_pos.mpr (by norm_num)
  have hs10lt : Real.sqrt 10 < 159 / 50 := by
    rw [Real.sqrt_lt' (by norm_num)]
    norm_num
  apply (lt_div_iff₀ (by positivity : 0 < 50 * Real.sqrt 10)).2
  nlinarith

lemma actual_deriv_first_eq :
    ofLp ((ofLp (radialRetentionDeriv 1 actualX5 actualV5)) 0) 0 =
      sharpJacobian 0 := by
  have hi : ((innerSL ℝ) actualX5) actualV5 = -1 / 3 := by
    simpa using actual_inner_eq
  simp only [radialRetentionDeriv, radialDeriv, sub_apply,
    smul_apply, ContinuousLinearMap.id_apply,
    ContinuousLinearMap.smulRight_apply]
  rw [actualX5_norm_sq, hi]
  simp [actualX5, actualV5, embedScalar5, sharpJacobian, sharpNormSq_eq,
    sharpDot_eq]
  ring

lemma actualV5_blockSup_eq : ‖ofLp actualV5‖ = 1 := by
  change ‖fun i : Fin 5 ↦ toLp 2 (fun j : Fin 2 ↦ if j = 0 then sharpV i else 0)‖ = 1
  apply le_antisymm
  · rw [pi_norm_le_iff_of_nonneg (by norm_num)]
    intro i
    fin_cases i <;> norm_num [sharpV, PiLp.norm_eq_of_L2, Fin.sum_univ_succ]
  · have h := norm_le_pi_norm (ofLp actualV5) (0 : Fin 5)
    norm_num [actualV5, embedScalar5, sharpV, PiLp.norm_eq_of_L2,
      Fin.sum_univ_succ] at h
    exact h

/-- The actual `radialRetentionDeriv` at the embedded five-block state expands
the manuscript's block-sup norm in the embedded direction. -/
theorem actual_five_block_deriv_expands :
    ‖ofLp actualV5‖ < ‖ofLp (radialRetentionDeriv 1 actualX5 actualV5)‖ := by
  rw [actualV5_blockSup_eq]
  have hblock := norm_le_pi_norm
    (ofLp (radialRetentionDeriv 1 actualX5 actualV5)) (0 : Fin 5)
  have hcoord :
      |ofLp ((ofLp (radialRetentionDeriv 1 actualX5 actualV5)) 0) 0| ≤
        ‖(ofLp (radialRetentionDeriv 1 actualX5 actualV5)) 0‖ := by
    simpa [dist_zero_right, Real.norm_eq_abs] using
      (PiLp.dist_apply_le
        ((ofLp (radialRetentionDeriv 1 actualX5 actualV5)) 0) 0 (0 : Fin 2))
  have hval : sharpJacobian 0 ≤
      ‖(ofLp (radialRetentionDeriv 1 actualX5 actualV5)) 0‖ := by
    rw [← actual_deriv_first_eq]
    exact (le_abs_self _).trans hcoord
  exact sharpJacobian_first_gt_one.trans_le (hval.trans hblock)

theorem actualX5_hasFDerivAt :
    HasFDerivAt (radialRetention (g := 5) 1)
      (radialRetentionDeriv 1 actualX5) actualX5 :=
  hasFDerivAt_radialRetention 1 (by norm_num) actualX5

abbrev BlockSup5 := (i : Fin 5) → EuclideanSpace ℝ (Fin 2)

noncomputable def actualRsup (z : BlockSup5) : BlockSup5 :=
  ofLp (radialRetention (g := 5) 1 (toLp 2 z))

noncomputable def actualRsupDeriv : BlockSup5 →L[ℝ] BlockSup5 :=
  (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).toContinuousLinearMap.comp
    ((radialRetentionDeriv 1 actualX5).comp
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).symm.toContinuousLinearMap)

theorem actualRsup_hasFDerivAt :
    HasFDerivAt actualRsup actualRsupDeriv (ofLp actualX5) := by
  have hin : HasFDerivAt
      (fun z : BlockSup5 ↦ toLp 2 z)
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).symm.toContinuousLinearMap
      (ofLp actualX5) :=
    (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).symm.hasFDerivAt
  have hmid := actualX5_hasFDerivAt.comp (ofLp actualX5) hin
  have hout : HasFDerivAt
      (fun z : Blocks 5 ↦ ofLp z)
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).toContinuousLinearMap
      (radialRetention 1 actualX5) :=
    (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin 5 ↦ EuclideanSpace ℝ (Fin 2))).hasFDerivAt
  exact hout.comp (ofLp actualX5) hmid

theorem actualRsupDeriv_expands :
    ‖ofLp actualV5‖ < ‖actualRsupDeriv (ofLp actualV5)‖ := by
  change ‖ofLp actualV5‖ <
    ‖ofLp (radialRetentionDeriv 1 actualX5 actualV5)‖
  exact actual_five_block_deriv_expands

/-- The actual five-block radial retention, expressed on the block-sup normed
space, is not globally non-expansive. -/
theorem actualRsup_not_one_lipschitz : ¬ LipschitzWith 1 actualRsup := by
  intro hlip
  have hop : ‖fderiv ℝ actualRsup (ofLp actualX5)‖ ≤ (1 : ℝ) :=
    norm_fderiv_le_of_lipschitz ℝ hlip
  rw [actualRsup_hasFDerivAt.fderiv] at hop
  have happ := actualRsupDeriv.le_opNorm (ofLp actualV5)
  rw [actualV5_blockSup_eq] at happ
  have : ‖actualRsupDeriv (ofLp actualV5)‖ ≤ 1 := by nlinarith
  have hex := actualRsupDeriv_expands
  rw [actualV5_blockSup_eq] at hex
  linarith

/-- The explicit Jacobian witness expands the five-block supremum norm. -/
theorem five_block_radial_jacobian_expands :
    scalarBlockSup sharpV < scalarBlockSup sharpJacobian := by
  rw [sharpV_blockSup_eq]
  have habs : sharpJacobian 0 ≤ |sharpJacobian 0| := le_abs_self _
  have hcoord : |sharpJacobian 0| ≤ ‖sharpJacobian‖ :=
    norm_le_pi_norm sharpJacobian (0 : Fin 5)
  exact sharpJacobian_first_gt_one.trans_le (habs.trans hcoord)

/-- Zero padding embeds the five-block witness into every larger block count. -/
noncomputable def padFive {g : ℕ} (v : Fin 5 → ℝ) (i : Fin g) : ℝ :=
  if h : i.val < 5 then v ⟨i.val, h⟩ else 0

theorem padFive_first {g : ℕ} (hg : 5 ≤ g) (v : Fin 5 → ℝ) :
    padFive v ⟨0, lt_of_lt_of_le (by norm_num) hg⟩ = v 0 := by simp [padFive]

theorem padFive_norm_eq {g : ℕ} (hg : 5 ≤ g) (v : Fin 5 → ℝ) :
    ‖padFive (g := g) v‖ = ‖v‖ := by
  apply le_antisymm
  · rw [pi_norm_le_iff_of_nonneg (norm_nonneg v)]
    intro i
    simp only [padFive]
    split_ifs with h
    · exact norm_le_pi_norm v ⟨i.val, h⟩
    · simp
  · have i0 : Fin g := ⟨0, by omega⟩
    rw [pi_norm_le_iff_of_nonneg (norm_nonneg _)]
    intro i
    let j : Fin g := ⟨i.val, lt_of_lt_of_le i.isLt hg⟩
    have hj : padFive (g := g) v j = v i := by simp [padFive, j]
    rw [← hj]
    exact norm_le_pi_norm (padFive (g := g) v) j

/-- The expanding witness persists in every dimension with at least five blocks. -/
theorem five_block_witness_embeds {g : ℕ} (hg : 5 ≤ g) :
    ‖padFive (g := g) sharpV‖ < ‖padFive (g := g) sharpJacobian‖ := by
  rw [padFive_norm_eq hg sharpV, padFive_norm_eq hg sharpJacobian]
  exact five_block_radial_jacobian_expands

#print axioms sharpNormSq_eq
#print axioms sharpDot_eq
#print axioms actualX5_norm_sq
#print axioms actual_inner_eq
#print axioms sharpV_blockSup_eq
#print axioms sharpJacobian_first_eq
#print axioms sharpJacobian_first_gt_one
#print axioms actual_deriv_first_eq
#print axioms actual_five_block_deriv_expands
#print axioms actualX5_hasFDerivAt
#print axioms actualRsup_hasFDerivAt
#print axioms actualRsupDeriv_expands
#print axioms actualRsup_not_one_lipschitz
#print axioms five_block_radial_jacobian_expands
#print axioms padFive_norm_eq
#print axioms five_block_witness_embeds

end NSI
