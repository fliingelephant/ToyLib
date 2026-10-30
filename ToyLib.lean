import Mathlib
import LeanBlame.Attr

/-!
# Basic notions of finite-dimensional quantum information

This file defines density operators, quantum channels, spectral gaps of
Hamiltonians, and separable states of bipartite systems.

## Conventions

* All Hilbert spaces are finite-dimensional. An operator on `ℂ^d` is a
  `d × d` complex matrix written in the computational basis `|0⟩, …, |d-1⟩`.
* A bipartite system `ℂ^m ⊗ ℂ^n` (first subsystem `A`, second subsystem `B`)
  uses the product basis `|i⟩ ⊗ |a⟩`, indexed by pairs `(i, a) : Fin m × Fin n`.
* `A.PosSemidef` (positive semidefinite) means that `A` is Hermitian and
  `⟨v, A v⟩ ≥ 0` for every vector `v`.

## Lean notation used in this file

* `open scoped ComplexOrder` turns on a partial order on `ℂ`:
  `z ≤ w` means `re z ≤ re w` and `im z = im w`. So `0 ≤ z` means that `z` is a
  nonnegative real number. `PosSemidef` needs this order to state `0 ≤ ⟨v, A v⟩`.
* `open scoped Kronecker` turns on the notation `A ⊗ₖ B` for the Kronecker
  product of matrices.
* `Mat m →ₗ[ℂ] Mat n` is the type of `ℂ`-linear maps from `Mat m` to `Mat n`.
* `Matrix.of f` is the matrix with entries `f i j`.
* `sInf S` is the infimum of a set `S` of real numbers. For a finite nonempty set
  it is the minimum.
* `Option ℝ` is the type whose values are either `some x` (with `x : ℝ`) or `none`.
  It is used for a quantity that may not exist.
* `(p i : ℂ) • A` views the real number `p i` as a complex number and multiplies
  the matrix `A` by it.
-/

namespace QuantumInformation

noncomputable section

open scoped ComplexOrder Kronecker

universe u

variable {m n : ℕ}

/-- Linear operators on the Hilbert space `ℂ^ι`, represented as complex square
matrices whose rows and columns are indexed by `ι`. -/
abbrev Operator (ι : Type u) := Matrix ι ι ℂ

/-- Linear operators on the `d`-dimensional Hilbert space `ℂ^d`. -/
abbrev Mat (d : ℕ) := Operator (Fin d)

/-- Linear operators on the bipartite Hilbert space `ℂ^m ⊗ ℂ^n`, in the product
basis `|i⟩ ⊗ |a⟩` indexed by `(i, a) : Fin m × Fin n`. -/
abbrev BipartiteMat (m n : ℕ) := Operator (Fin m × Fin n)

/-- `ρ` is a density operator (a quantum state): `ρ` is positive semidefinite
and has trace one, `Tr ρ = 1`. -/
def IsDensity {ι : Type u} [Fintype ι] [DecidableEq ι]
    (ρ : Operator ι) : Prop :=
  ρ.PosSemidef ∧ ρ.trace = 1

/-- The linear map `Φ` is trace-preserving: `Tr (Φ A) = Tr A` for every
operator `A`, not only for density operators. -/
def IsTracePreserving (Φ : Mat m →ₗ[ℂ] Mat n) : Prop :=
  ∀ A : Mat m, (Φ A).trace = A.trace

/-- `Φ` is a quantum channel from `ℂ^m` to `ℂ^n`: a completely positive and
trace-preserving (CPTP) linear map.

Complete positivity means that for every ancilla dimension `k`, the extended map
`id_k ⊗ Φ` sends positive semidefinite operators on `ℂ^k ⊗ ℂ^m` to positive
semidefinite operators on `ℂ^k ⊗ ℂ^n`. Physically, `Φ` must output a valid state
even when it acts on one part of a larger system that is entangled with the rest.

The assumptions `NeZero m` and `NeZero n` ensure that both spaces are nonzero,
so that density operators exist on both sides. -/
@[trusted "4ed1b658-9b1a-46a3-9d4f-dc2794946003"]
def IsQuantumChannel [NeZero m] [NeZero n] (Φ : Mat m →ₗ[ℂ] Mat n) : Prop :=
  IsTracePreserving Φ ∧
  ∀ ρ : Mat m, IsDensity ρ → IsDensity (Φ ρ)

/-- A Hamiltonian on `ℂ^d`: a Hermitian `d × d` complex matrix, with `d ≥ 1`. -/
structure Hamiltonian (d : ℕ) where
  /-- The Hilbert space is nonzero, so the spectrum is nonempty. -/
  dim_pos : 0 < d
  /-- The matrix of the Hamiltonian in the computational basis. -/
  matrix : Mat d
  /-- The matrix is Hermitian, so all its eigenvalues are real. -/
  hermitian : matrix.IsHermitian

/-- The spectrum of `H`: the set of its distinct eigenvalues (energy levels).
It is a finite, nonempty set of real numbers.

`H.hermitian.eigenvalues : Fin d → ℝ` lists the eigenvalues with multiplicity;
taking its range removes repetitions. -/
def energySpectrum {d : ℕ} (H : Hamiltonian d) : Set ℝ :=
  Set.range H.hermitian.eigenvalues

/-- The ground-state energy `E₀` of `H`: its smallest eigenvalue. -/
def groundEnergy {d : ℕ} (H : Hamiltonian d) : ℝ :=
  sInf (energySpectrum H)

/-- The excited energies of `H`: the eigenvalues strictly greater than the
ground-state energy `E₀`. -/
def excitedEnergies {d : ℕ} (H : Hamiltonian d) : Set ℝ :=
  {E ∈ energySpectrum H | groundEnergy H < E}

open Classical in
/-- The spectral gap of `H`: `E₁ - E₀`, where `E₀` is the ground-state energy and
`E₁` is the smallest eigenvalue strictly greater than `E₀`.

The ground state may be degenerate. The gap is measured above the whole
ground-state eigenspace, so it is always positive when it exists.
The result is `none` when `H` has only one distinct eigenvalue (that is, `H` is a
multiple of the identity), because then there is no excited energy. -/
def spectralGap {d : ℕ} (H : Hamiltonian d) : Option ℝ :=
  if (excitedEnergies H).Nonempty then
    some (sInf (excitedEnergies H) - groundEnergy H)
  else
    none

/-- A family of Hamiltonians `H N`, indexed by the system size `N` and acting on
`ℂ^(d N)`, is gapped (has a uniform spectral gap) if there exist a constant
`Δ > 0` and a threshold `N₀` such that for every `N ≥ N₀`, the spectral gap of
`H N` exists and is at least `Δ`.

The constant `Δ` must not depend on `N`. The threshold `N₀` allows finitely many
small system sizes to be ignored, since only the behavior as `N → ∞` matters. -/
@[trusted "6d8ce80f-c5b2-4c39-a8f0-3e71852a1132"]
def IsGapped {d : ℕ → ℕ} (H : (N : ℕ) → Hamiltonian (d N)) : Prop :=
  ∃ Δ : ℝ, 0 < Δ ∧
    ∀ N : ℕ, ∃ N₀ : ℕ, N₀ ≤ N →
      ∃ g : ℝ, spectralGap (H N) = some g ∧ Δ ≤ g

/-- `p` is a probability distribution on `k` outcomes: `p i ≥ 0` for every `i`,
and `∑ i, p i = 1`. -/
def IsProbabilityVector {k : ℕ} (p : Fin k → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i : Fin k, p i) = 1

/-- The tensor product `A ⊗ B` of an operator `A` on `ℂ^m` and an operator `B` on
`ℂ^n`, as an operator on `ℂ^m ⊗ ℂ^n` (the Kronecker product of matrices).
Its entries are `(A ⊗ B)_{(i,a),(j,b)} = A_{ij} B_{ab}`. -/
def tensor (A : Mat m) (B : Mat n) : BipartiteMat m n :=
  A ⊗ₖ B

/-- The partial trace over the second subsystem, `Tr_B`, which maps operators on
`ℂ^m ⊗ ℂ^n` to operators on `ℂ^m`. Its entries are
`(Tr_B ρ)_{ij} = ∑_a ρ_{(i,a),(j,a)}`. On product operators,
`Tr_B (A ⊗ B) = (Tr B) A`. -/
def partialTraceRight (ρ : BipartiteMat m n) : Mat m :=
  Matrix.of fun i j => ∑ a : Fin n, ρ (i, a) (j, a)

/-- The partial trace over the first subsystem, `Tr_A`, which maps operators on
`ℂ^m ⊗ ℂ^n` to operators on `ℂ^n`. Its entries are
`(Tr_A ρ)_{ab} = ∑_i ρ_{(i,a),(i,b)}`. On product operators,
`Tr_A (A ⊗ B) = (Tr A) B`. -/
def partialTraceLeft (ρ : BipartiteMat m n) : Mat n :=
  Matrix.of fun a b => ∑ i : Fin m, ρ (i, a) (i, b)

/-- The partial transpose `ρ^{T_B}` with respect to the second subsystem, taken in
the computational basis. Its entries are `(ρ^{T_B})_{(i,a),(j,b)} = ρ_{(i,b),(j,a)}`.
On product operators, `(A ⊗ B)^{T_B} = A ⊗ Bᵀ`. -/
def partialTranspose (ρ : BipartiteMat m n) : BipartiteMat m n :=
  Matrix.of fun (i, a) (j, b) => ρ (i, b) (j, a)

/-- `ρ` is a PPT (positive partial transpose) state: `ρ` is a density operator and
its partial transpose `ρ^{T_B}` is positive semidefinite. By the Peres criterion,
every separable state is PPT. -/
def IsPPT (ρ : BipartiteMat m n) : Prop :=
  IsDensity ρ ∧ (partialTranspose ρ).PosSemidef

/-- `ω` is a separable state on `ℂ^m ⊗ ℂ^n`: a finite convex combination of
product states,
`ω = ∑ i, p i • (ρ i ⊗ σ i)`,
where `p` is a probability distribution on `k` outcomes, each `ρ i` is a density
operator on `ℂ^m`, and each `σ i` is a density operator on `ℂ^n`.

A state that is not separable is called entangled. In finite dimensions a finite
number `k` of terms always suffices (Carathéodory's theorem). -/
@[trusted "3703585c-f15a-4f3d-8ebb-792a899b12ec"]
def IsSeparable (ω : BipartiteMat m n) : Prop :=
  ∃ k : ℕ, ∃ p : Fin k → ℝ, ∃ ρ : Fin k → Mat m, ∃ σ : Mat n,
    IsProbabilityVector p ∧
    (∀ i, IsDensity (ρ i)) ∧
    IsDensity σ ∧
    ω = ∑ i : Fin k, (p i : ℂ) • tensor (ρ i) σ


/-- The identity map is a quantum channel. -/
@[trusted "a30a4368-b5ed-402d-961f-fee1a7808c13"]
theorem isQuantumChannel_id [NeZero m] : IsQuantumChannel (LinearMap.id : Mat m →ₗ[ℂ] Mat m) :=
  ⟨fun _ => rfl, fun _ h => h⟩

/-- A constant family whose Hamiltonian has a positive spectral gap is gapped. -/
@[trusted "7de85122-a8c3-473c-b577-b65d55f00fe0"]
theorem isGapped_const {d : ℕ} (H : Hamiltonian d) {g : ℝ} (hg : spectralGap H = some g) (hpos : 0 < g) :
    IsGapped (fun _ : ℕ => H) :=
  ⟨g, hpos, fun _ => ⟨0, fun _ => ⟨g, hg, le_rfl⟩⟩⟩

/-- The tensor product `ρ ⊗ σ` of two density operators is separable. -/
@[trusted "0e50d074-7699-4dc9-a76f-5b5dcf42db16"]
theorem isSeparable_tensor {ρ : Mat m} {σ : Mat n} (hρ : IsDensity ρ) (hσ : IsDensity σ) :
    IsSeparable (tensor ρ σ) :=
  ⟨1, fun _ => 1, fun _ => ρ, σ, ⟨fun _ => zero_le_one, by simp⟩, fun _ => hρ, hσ, by simp⟩

end

end QuantumInformation
