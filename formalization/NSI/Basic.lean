import Mathlib

namespace NSI

variable {X : Type*} [PseudoMetricSpace X]

/-- Composition of a non-expansive retention map with a q-Lipschitz
aggregation map is q-Lipschitz. This is the abstract outer shell of the
fixed-input NSI update; the concrete retention proof is a separate target. -/
theorem completeUpdate_lipschitz
    (q : NNReal) (R L : X → X)
    (hR : LipschitzWith 1 R) (hL : LipschitzWith q L) :
    LipschitzWith q (L ∘ R) := by
  simpa using hL.comp hR

/-- A shared contraction makes differences of consecutive iterates decay. -/
theorem increment_step
    (q : NNReal) (T : X → X) (hT : LipschitzWith q T)
    (h₀ h₁ : X) :
    dist (T h₁) (T h₀) ≤ q * dist h₁ h₀ :=
  hT.dist_le_mul h₁ h₀

/-- One perturbation step for an inexact trajectory under a shared contraction. -/
theorem inexact_step
    (q : NNReal) (T : X → X) (hT : LipschitzWith q T)
    (exact approx nextApprox : X) (ε : ℝ)
    (hnext : dist nextApprox (T approx) ≤ ε) :
    dist nextApprox (T exact) ≤ ε + q * dist approx exact := by
  calc
    dist nextApprox (T exact)
        ≤ dist nextApprox (T approx) + dist (T approx) (T exact) := dist_triangle _ _ _
    _ ≤ ε + q * dist approx exact := add_le_add hnext (hT.dist_le_mul approx exact)

/-- The scalar recurrence used for the direct state-magnitude bound. -/
theorem state_step
    (q d s next : ℝ) (hnext : next ≤ q * (d + s)) :
    next ≤ q * d + q * s := by
  simpa [mul_add] using hnext

#print axioms completeUpdate_lipschitz
#print axioms increment_step
#print axioms inexact_step
#print axioms state_step

end NSI
