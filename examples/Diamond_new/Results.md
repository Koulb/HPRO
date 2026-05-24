# Diamond_new Band-RI Reconstruction: Analysis Results

## Problem

The Band-RI real-space reconstruction (`H_ij(R) = sum_nk w_k A+_{i0nk} eps_{nk} A_{jRnk}`)
gives catastrophically bad band structure when using a 4x4x4 nscf k-grid (this folder),
but works well with a 6x6x6 k-grid (see `../Diamond_new_k6/`).

## Root cause: R-space truncation from insufficient k-grid

### Comparison with Diamond_new_k6

| Aspect | Diamond_new | Diamond_new_k6 |
|--------|------------|----------------|
| nscf k-grid | 4x4x4 (symlink to Diamond/nscf) | 6x6x6 (own nscf directory) |
| bands, scf, aobasis | symlink to Diamond/ | symlink to Diamond/ |
| hamiltonians.h5, overlaps.h5 | symlink to Diamond/ | symlink to Diamond/ |
| Python scripts | identical | identical |

The **only** difference is the nscf k-grid density. All scripts
(`compare_reconstruction_real.py`, `diag.py`, `plotband_real.py`) are byte-for-byte
identical between the two folders.

### Why the 4x4x4 grid fails

1. **R-space truncation**: The original reconstruction has 105 R-vectors (with |R_i|
   up to 3). A 4x4x4 grid gives canonical R-range [-2, 1], so only 54/105 R-vectors
   fit. The remaining 51 are zeroed out in `hamiltonians_ri.h5`.

2. **Catastrophic Fourier interpolation**: With only 64 canonical R-vectors in [-2,1]^3,
   the FT `H(k) = sum_R H(R) exp(2pi i k.R)` at arbitrary band-path k-points produces
   eigenvalues as low as **-7246 eV** (33 out of 71 k-points are affected). This is due
   to Gibbs-like oscillations from the truncated Fourier series, amplified by the
   high-energy bands (~100+ eV) in the RI sum.

3. **Direct k-space formula is fine**: `compare_reconstruction_real.py` uses
   `H(k) = A†(k) diag(ε) A(k)` directly (no R-space roundtrip) and gives
   MAE = 46.8 meV (k4) vs 46.0 meV (k6) — essentially identical. The problem is
   strictly in the R→k→diag pipeline.

4. **Why 6x6x6 works**: Canonical range [-3, 2] encompasses all 105 original R-vectors.
   No information is lost, Fourier interpolation is well-behaved.

### Numerical evidence

| Metric | Diamond_new (4x4x4) | Diamond_new_k6 (6x6x6) |
|--------|---------------------|------------------------|
| Canonical R-vectors | 64 | 216 |
| Orig R-vectors in range | 54/105 | 105/105 |
| Band-RI MAE (direct k-space) | 46.8 meV | 46.0 meV |
| Band-RI MAE (R-roundtrip, plotband) | 459,198 meV | 284 meV |
| Min eigenvalue in eig_ri.dat | -7246 eV | -8.02 eV |
| Problematic k-points | 33/71 | 0/71 |

## Rule of thumb

For the Band-RI R-space reconstruction to work correctly, the nscf k-grid dimension
N_i must satisfy:

    N_i >= 2 * max(|R_i|) + 1

For Diamond with max(|R_i|) = 3, this gives N_i >= 7, so a 6x6x6 grid is the minimum
that captures all R-vectors (since [-3, 2] covers |R_i| <= 3).

## Additional consideration: real-space hermitianization

See `../Diamond_new_k6/Results.md` for details on the secondary issue of
real-space vs k-space hermitianization, which was also addressed in `diag.py`.

---

## Alignment investigation (Feb 2026)

### Why Diamond/plotband.py reports ~1110 meV while Diamond_new_k6 reports ~282 meV

The two examples share **byte-for-byte identical** hamiltonians.h5, overlaps.h5, and
eig.dat files (verified via md5sum). The charge density used for nscf in both cases
comes from the same scf directory.

The entire MAE difference comes from the **energy alignment method**:
- Diamond/plotband.py: Fermi-aligned (shifts both QE and reconstruction by E_Fermi)
- Diamond_new_k6/plotband_real.py: VBM-aligned (shifts by max of band 4 independently)

With VBM alignment, the "Original" reconstruction MAE for 4 bands is ~282 meV in both.

### Error decomposition (VBM-aligned, 4 bands, band-path k-points)

| Method | MAE (meV) | What it measures |
|--------|-----------|------------------|
| Direct k-space Band-RI | ~52 | LCAO basis projection error only |
| R-space Original reconstruction | ~282 | Projection + V_nl + numerical errors |
| R-space Band-RI reconstruction | ~284 | Same as above (different route, same H(R)) |

The ~230 meV gap between direct k-space (~52 meV) and R-space (~282 meV) is dominated
by the nonlocal pseudopotential (V_nl) error in the original reconstruction, which is
computed with `ecutwfn=30` (vs QE's `ecutwfc=60`).

### Test plan: ecutwfn convergence

The original reconstruction uses `ecutwfn=30` Ry for computing overlap integrals and
V_nl projector overlaps. QE uses `ecutwfc=60` Ry. Increasing `ecutwfn` to 60 may
reduce the V_nl error and close the gap between 52 and 282 meV.

Similarly, the Band-RI A-matrix computation uses `ecut=30` in `calc_FT_kg_orb_spcs`.
Testing with `ecut=60` may improve A = <psi|phi> accuracy.
