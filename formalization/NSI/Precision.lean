import NSI.Extensions

namespace NSI
open Finset Function

theorem precisionBound_closed (q : ℝ) (ε : ℕ → ℝ) (e₀ : ℝ) (k : ℕ) :
    precisionBound q ε e₀ k =
      q^k * e₀ + ∑ j ∈ range k, q^(k-1-j) * ε j := by
  induction k with
  | zero => simp [precisionBound]
  | succ k ih =>
      have hs : q * (∑ j ∈ range k, q^(k-1-j) * ε j) =
          ∑ j ∈ range k, q^(k-j) * ε j := by
        rw [mul_sum]
        apply sum_congr rfl
        intro j hj
        have he : k-j = (k-1-j)+1 := by have := mem_range.mp hj; omega
        rw [he, pow_succ]
        ring
      rw [precisionBound, ih, sum_range_succ]
      simp only [Nat.add_sub_cancel, Nat.sub_self, pow_zero, one_mul]
      rw [mul_add, hs, pow_succ]
      ring

theorem finite_precision_general_closed
    (q : ℝ) (ε e : ℕ → ℝ) (e₀ : ℝ) (hq : 0 ≤ q)
    (hzero : e 0 ≤ e₀)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + ε k) (k : ℕ) :
    e k ≤ q^k * e₀ + ∑ j ∈ range k, q^(k-1-j) * ε j := by
  rw [← precisionBound_closed]
  exact finite_precision_le_precisionBound q ε e e₀ hq hzero hstep k

/-- The complete error bound for an inexact orbit of one shared map. -/
theorem inexact_orbit_bound
    {E : Type*} [PseudoMetricSpace E]
    (q : NNReal) (T : E → E) (hT : LipschitzWith q T)
    (x₀ : E) (a : ℕ → E) (ε : ℕ → ℝ)
    (hloc : ∀ k, dist (a (k+1)) (T (a k)) ≤ ε k) (K : ℕ) :
    dist (a K) ((T^[K]) x₀) ≤
      (q:ℝ)^K * dist (a 0) x₀ +
        ∑ j ∈ range K, (q:ℝ)^(K-1-j) * ε j := by
  apply finite_precision_general_closed (q:ℝ) ε
    (fun k => dist (a k) ((T^[k]) x₀)) (dist (a 0) x₀) q.coe_nonneg
  · simp
  · intro k
    rw [iterate_succ_apply']
    calc
      dist (a (k+1)) (T ((T^[k]) x₀)) ≤
          dist (a (k+1)) (T (a k)) + dist (T (a k)) (T ((T^[k]) x₀)) :=
        dist_triangle _ _ _
      _ ≤ ε k + q * dist (a k) ((T^[k]) x₀) :=
        add_le_add (hloc k) (hT.dist_le_mul _ _)
      _ = _ := add_comm _ _

#print axioms precisionBound_closed
#print axioms finite_precision_general_closed
#print axioms inexact_orbit_bound
end NSI
