# Diamond Band-RI Reconstruction: Investigation Results

## Problem

The new Band-RI reconstruction (`H_ij(R) = sum_nk w_k A+_{i0nk} eps_{nk} A_{jRnk}`) gave
poor results for Diamond (noisy/scattered conduction bands) while MoS2 worked well.

## Root Cause (two issues)

### 1. Insufficient k-grid density (primary)

The original 4x4x4 nscf k-grid (64 k-points) was too coarse for Diamond's 3D FCC structure.
Only 54 of 105 R-vectors fell within the canonical range `[-N_i//2, (N_i-1)//2]`;
the remaining 51 were zeroed out, losing significant Hamiltonian information.

MoS2 (2D, large vacuum along z) has all R-vectors within range even with its original grid
because R_z = 0 always.

**Fix**: Use 6x6x6 k-grid (216 k-points). Now all 105 R-vectors are in range,
and the canonical set has 216 vectors.

### 2. Real-space hermitianization (secondary)

`LCAODiagKernel.load_deeph_mats()` automatically calls `matH.hermitianize()` which
averages `H(R)` with `H(-R)^dagger` in real space. For the Band-RI Hamiltonian,
the finite band summation (200 bands) introduces a non-Hermiticity error of ~0.003 Ha (~85 meV).
Hermitianizing in real space distorts the H(R) matrices.

**Fix**: Load `hamiltonians_ri.h5` directly via `load_deeph_HS()` (skipping hermitianize),
then hermitianize only in k-space: `Hk = 0.5*(Hk + Hk^dagger)` before diagonalization.
This is physically correct since H(k) must be Hermitian, while H(R) need not be when
the band sum is incomplete.

## Results (6x6x6 k-grid, k-space hermitianization)

MAE vs DFT (meV), VBM-aligned, 71 k-points along Gamma-X-W-L-Gamma:

| Bands | Original | Band-RI |
|-------|----------|---------|
| 4     | 282      | 284     |
| 8     | 311      | 310     |
| 16    | 3076     | 2736    |

Both methods are now comparable. Band-RI is even slightly better for 16 bands.

## H(R) comparison (Band-RI vs Original, 200 bands)

- H(R=0) relative error: 1.43%
- H(R=nearest neighbor) relative error: ~2.1%
- H(R=0) convergence: 38% at 10 bands -> 1.4% at 200 bands
- Max imaginary part ratio: 7.1e-7 (negligible)
- Roundtrip error (save/reload): 2.4e-16 (machine precision)

## Key files modified

- `nscf/pw.in` — 6x6x6 uniform k-grid (216 k-points), 200 bands
- `reconstruction/aohamiltonian/diag.py` — direct loading for Band-RI (no hermitianize)
- `reconstruction/aohamiltonian/compare_reconstruction_real.py` — unchanged (paths are relative)
- `reconstruction/aohamiltonian/plotband_real.py` — unchanged

## Lessons learned

1. For 3D materials, the nscf k-grid must be dense enough so all R-vectors of the
   original reconstruction fall within the canonical range of the DFT. Rule of thumb:
   grid dimension N_i >= 2*max(|R_i|) + 1.

2. Do NOT hermitianize Band-RI H(R) in real space. The incomplete band sum makes
   H(R) != H(-R)^dagger. Hermitianize only H(k) before the generalized eigenvalue problem.

3. MoS2 worked despite using `LCAODiagKernel` (which hermitianizes) because its
   H(R) non-Hermiticity was much smaller (~1e-6 Ha vs ~3e-3 Ha for Diamond).

## Cross-reference

For a detailed comparison of the 4x4x4 (Diamond_new) vs 6x6x6 (this folder) k-grids,
including numerical evidence of R-space truncation and catastrophic Fourier interpolation,
see `../Diamond_new/Results.md`. The 6x6x6 grid was chosen precisely to encompass all
105 R-vectors of the original reconstruction (canonical range [-3, 2] covers max |R_i| = 3).
