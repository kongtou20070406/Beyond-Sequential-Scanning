import NSI.Sharpness

namespace NSI

open Finset
open WithLp

/-! The actual five-block witness, padded by zero blocks.  Writing the group
size as `5 + k` covers every natural group size at least five. -/

noncomputable def padBlocks {k : ℕ} (x : Blocks 5) : Blocks (5 + k) :=
  toLp 2 (Fin.addCases (ofLp x) (fun _ : Fin k ↦ 0))

noncomputable def actualXAll (k : ℕ) : Blocks (5 + k) := padBlocks actualX5
noncomputable def actualVAll (k : ℕ) : Blocks (5 + k) := padBlocks actualV5

lemma padBlocks_norm_sq {k : ℕ} (x : Blocks 5) :
    ‖padBlocks (k := k) x‖ ^ 2 = ‖x‖ ^ 2 := by
  simp [padBlocks, PiLp.norm_eq_of_L2, Fin.sum_univ_add]

lemma padBlocks_inner {k : ℕ} (x y : Blocks 5) :
    inner ℝ (padBlocks (k := k) x) (padBlocks y) = inner ℝ x y := by
  simp [padBlocks, PiLp.inner_apply, Fin.sum_univ_add]

lemma actualXAll_norm_sq (k : ℕ) : ‖actualXAll k‖ ^ 2 = 1 / 9 := by
  rw [actualXAll, padBlocks_norm_sq, actualX5_norm_sq]

lemma actualXAll_inner (k : ℕ) :
    inner ℝ (actualXAll k) (actualVAll k) = -1 / 3 := by
  rw [actualXAll, actualVAll, padBlocks_inner, actual_inner_eq]

lemma actualVAll_first (k : ℕ) :
    ofLp ((ofLp (actualVAll k)) (Fin.castAdd k 0)) 0 = 1 := by
  simp [actualVAll, padBlocks, actualV5, embedScalar5, sharpV]

lemma actual_deriv_all_first_eq (k : ℕ) :
    ofLp ((ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
      (Fin.castAdd k 0)) 0 = sharpJacobian 0 := by
  have hi : ((innerSL ℝ) (actualXAll k)) (actualVAll k) = -1 / 3 := by
    simpa using actualXAll_inner k
  simp only [radialRetentionDeriv, radialDeriv, sub_apply,
    smul_apply, ContinuousLinearMap.id_apply,
    ContinuousLinearMap.smulRight_apply]
  rw [actualXAll_norm_sq, hi]
  simp [actualXAll, actualVAll, padBlocks, actualX5, actualV5, embedScalar5,
    sharpJacobian, sharpNormSq_eq, sharpDot_eq, sharpX, sharpV]
  ring

lemma actualVAll_blockSup_eq (k : ℕ) : ‖ofLp (actualVAll k)‖ = 1 := by
  apply le_antisymm
  · rw [pi_norm_le_iff_of_nonneg (by norm_num)]
    intro i
    refine Fin.addCases ?_ ?_ i
    · intro j
      simpa [actualVAll, padBlocks] using
        (show ‖(ofLp actualV5) j‖ ≤ 1 from
          (norm_le_pi_norm (ofLp actualV5) j).trans_eq actualV5_blockSup_eq)
    · intro j
      simp [actualVAll, padBlocks]
  · have h := norm_le_pi_norm (ofLp (actualVAll k)) (Fin.castAdd k 0)
    have hc : ‖(ofLp (actualVAll k)) (Fin.castAdd k 0)‖ = 1 := by
      simp [actualVAll, padBlocks, actualV5, embedScalar5, sharpV,
        PiLp.norm_eq_of_L2]
    linarith

theorem actual_all_groups_deriv_expands (k : ℕ) :
    ‖ofLp (actualVAll k)‖ <
      ‖ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k))‖ := by
  rw [actualVAll_blockSup_eq]
  have hblock := norm_le_pi_norm
    (ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
    (Fin.castAdd k 0)
  have hcoord :
      |ofLp ((ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
          (Fin.castAdd k 0)) 0| ≤
        ‖(ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
          (Fin.castAdd k 0)‖ := by
    simpa [dist_zero_right, Real.norm_eq_abs] using
      (PiLp.dist_apply_le
        ((ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
          (Fin.castAdd k 0)) 0 (0 : Fin 2))
  have hval : sharpJacobian 0 ≤
      ‖(ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)))
        (Fin.castAdd k 0)‖ := by
    rw [← actual_deriv_all_first_eq]
    exact (le_abs_self _).trans hcoord
  exact sharpJacobian_first_gt_one.trans_le (hval.trans hblock)

abbrev BlockSupAll (k : ℕ) :=
  (i : Fin (5 + k)) → EuclideanSpace ℝ (Fin 2)

noncomputable def actualRsupAll (k : ℕ) (z : BlockSupAll k) : BlockSupAll k :=
  ofLp (radialRetention (g := 5 + k) 1 (toLp 2 z))

noncomputable def actualRsupDerivAll (k : ℕ) :
    BlockSupAll k →L[ℝ] BlockSupAll k :=
  (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).toContinuousLinearMap.comp
    ((radialRetentionDeriv 1 (actualXAll k)).comp
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).symm.toContinuousLinearMap)

lemma actualRsupAll_hasFDerivAt (k : ℕ) :
    HasFDerivAt (actualRsupAll k) (actualRsupDerivAll k)
      (ofLp (actualXAll k)) := by
  have hin : HasFDerivAt
      (fun z : BlockSupAll k ↦ toLp 2 z)
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).symm.toContinuousLinearMap
      (ofLp (actualXAll k)) :=
    (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).symm.hasFDerivAt
  have hmid := (hasFDerivAt_radialRetention 1 (by norm_num) (actualXAll k)).comp
    (ofLp (actualXAll k)) hin
  have hout : HasFDerivAt
      (fun z : Blocks (5 + k) ↦ ofLp z)
      (PiLp.continuousLinearEquiv 2 ℝ
        (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).toContinuousLinearMap
      (radialRetention 1 (actualXAll k)) :=
    (PiLp.continuousLinearEquiv 2 ℝ
      (fun _ : Fin (5 + k) ↦ EuclideanSpace ℝ (Fin 2))).hasFDerivAt
  exact hout.comp (ofLp (actualXAll k)) hmid

theorem actualRsupAll_not_one_lipschitz (k : ℕ) :
    ¬ LipschitzWith 1 (actualRsupAll k) := by
  intro hlip
  have hop : ‖fderiv ℝ (actualRsupAll k) (ofLp (actualXAll k))‖ ≤ (1 : ℝ) :=
    norm_fderiv_le_of_lipschitz ℝ hlip
  rw [(actualRsupAll_hasFDerivAt k).fderiv] at hop
  have happ := (actualRsupDerivAll k).le_opNorm (ofLp (actualVAll k))
  rw [actualVAll_blockSup_eq] at happ
  have heq :
      actualRsupDerivAll k (ofLp (actualVAll k)) =
        ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k)) := rfl
  rw [heq] at happ
  have hle :
      ‖ofLp (radialRetentionDeriv 1 (actualXAll k) (actualVAll k))‖ ≤ 1 := by
    nlinarith
  have hex := actual_all_groups_deriv_expands k
  rw [actualVAll_blockSup_eq] at hex
  linarith

#print axioms padBlocks_norm_sq
#print axioms padBlocks_inner
#print axioms actual_deriv_all_first_eq
#print axioms actual_all_groups_deriv_expands
#print axioms actualRsupAll_hasFDerivAt
#print axioms actualRsupAll_not_one_lipschitz

end NSI
