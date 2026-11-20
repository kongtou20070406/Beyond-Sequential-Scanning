import NSI.Retention
import Mathlib.Analysis.Calculus.MeanValue

namespace NSI

set_option maxHeartbeats 1000000

open WithLp

abbrev BlockSup (g : ℕ) := (i : Fin g) → EuclideanSpace ℝ (Fin 2)

noncomputable def blockEquiv (g : ℕ) : Blocks g ≃L[ℝ] BlockSup g :=
  PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin g => EuclideanSpace ℝ (Fin 2))

noncomputable def radialRetentionSup {g : ℕ} (alpha : ℝ) (z : BlockSup g) : BlockSup g :=
  blockEquiv g (radialRetention alpha ((blockEquiv g).symm z))

noncomputable def radialRetentionSupDeriv {g : ℕ} (alpha : ℝ) (z : BlockSup g) :
    BlockSup g →L[ℝ] BlockSup g :=
  (blockEquiv g).toContinuousLinearMap.comp
    ((radialRetentionDeriv alpha ((blockEquiv g).symm z)).comp
      (blockEquiv g).symm.toContinuousLinearMap)

theorem hasFDerivAt_radialRetentionSup {g : ℕ} (alpha : ℝ) (halpha : 0 ≤ alpha)
    (z : BlockSup g) :
    HasFDerivAt (radialRetentionSup alpha) (radialRetentionSupDeriv alpha z) z := by
  exact (blockEquiv g).toContinuousLinearMap.hasFDerivAt.comp z
    ((hasFDerivAt_radialRetention alpha halpha ((blockEquiv g).symm z)).comp z
      (blockEquiv g).symm.toContinuousLinearMap.hasFDerivAt)

theorem radialRetentionSupDeriv_norm_le_one {g : ℕ} [NeZero g]
    (alpha : ℝ) (halpha : 0 ≤ alpha) (hg4 : g ≤ 4) (z : BlockSup g) :
    ‖radialRetentionSupDeriv alpha z‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ zero_le_one
  intro v
  change ‖ofLp (radialRetentionDeriv alpha ((blockEquiv g).symm z)
    ((blockEquiv g).symm v))‖ ≤ 1 * ‖v‖
  convert radialRetentionDeriv_blockSup_le alpha halpha hg4
    ((blockEquiv g).symm z) ((blockEquiv g).symm v) using 1
  simp [blockEquiv]

theorem radialRetentionSup_nonexpansive {g : ℕ} [NeZero g]
    (alpha : ℝ) (halpha : 0 ≤ alpha) (hg4 : g ≤ 4) (x y : BlockSup g) :
    ‖radialRetentionSup alpha x - radialRetentionSup alpha y‖ ≤ ‖x - y‖ := by
  have hf : ∀ z ∈ (Set.univ : Set (BlockSup g)),
      HasFDerivWithinAt (radialRetentionSup alpha)
        (radialRetentionSupDeriv alpha z) Set.univ z := by
    intro z hz
    exact (hasFDerivAt_radialRetentionSup alpha halpha z).hasFDerivWithinAt
  have hb : ∀ z ∈ (Set.univ : Set (BlockSup g)),
      ‖radialRetentionSupDeriv alpha z‖ ≤ (1 : ℝ) := by
    intro z hz
    exact radialRetentionSupDeriv_norm_le_one alpha halpha hg4 z
  simpa using (convex_univ.norm_image_sub_le_of_norm_hasFDerivWithin_le hf hb
    (Set.mem_univ y) (Set.mem_univ x))

theorem radialRetention_blockSup_nonexpansive {g : ℕ} [NeZero g]
    (alpha : ℝ) (halpha : 0 ≤ alpha) (hg4 : g ≤ 4) (x y : Blocks g) :
    ‖ofLp (radialRetention alpha x - radialRetention alpha y)‖ ≤ ‖ofLp (x - y)‖ := by
  simpa [radialRetentionSup, blockEquiv, PiLp.coe_continuousLinearEquiv] using
    radialRetentionSup_nonexpansive alpha halpha hg4 (ofLp x) (ofLp y)

#print axioms radialRetentionSup_nonexpansive
#print axioms radialRetention_blockSup_nonexpansive

end NSI
