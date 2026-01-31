"""
Compare standard LCAO diagonalization with band-RI reconstruction.

This script compares:
1. Original H(k) from PW→AO projection (stored in hamiltonians.h5)
2. Band-RI reconstructed H_loc = A† @ diag(ε) @ A

Both matrices are in the localized (AO) basis.
"""

import numpy as np
from HPRO.lcaodiag import LCAODiagKernel
from HPRO.deephio import load_deeph_HS
from HPRO.constants import hartree2ev
from HPRO.mathutils import compute_local_h_band_ri

# K-path: Γ-K-M-Γ
kpts = [[0.000000000000, 0.000000000000, 0.000000000000],
        [0.333333333333, 0.333333333333, 0.000000000000],
        [0.500000000000, 0.000000000000, 0.000000000000],
        [0.000000000000, 0.000000000000, 0.000000000000]]
kptwts = [20, 10, 17, 1]
kptsymbols = ['Γ', 'K', 'M', 'Γ']

nbnd = 36

print("=" * 70)
print("Comparison: Original H(k) vs Band-RI H_loc")
print("=" * 70)

# Step 1: Standard diagonalization
print("\n[1] Running LCAO diagonalization...")
kernel = LCAODiagKernel()
kernel.setk(kpts, kptwts, kptsymbols)
kernel.load_deeph_mats('./')
kernel.diag(nbnd=nbnd, efermi=None)

nao = kernel.nao
print(f"    Number of AOs: {nao}")
print(f"    Number of bands: {nbnd}")

# Step 2: Load original H(k) and S(k) matrices
print("\n[2] Loading original H(k) and S(k) matrices from files...")
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Step 3: Band-RI convergence using the built-in method
print("\n[3] Band-RI convergence (built-in method)...")
print("    Formula: H_loc = A† @ diag(ε) @ A")
result = kernel.band_ri_convergence(nband_list=[9, 18, 27, 36])

# Step 4: Direct matrix comparison - H_orig(k) vs H_loc_bandRI(k)
print("\n" + "=" * 70)
print("[4] Matrix Comparison: H_orig(k) vs H_loc_bandRI(k)")
print("=" * 70)
print("\nBoth matrices are in the localized (AO) basis.")
print("H_orig(k) = original from PW→AO projection")
print("H_loc(k)  = A† @ diag(ε) @ A (band-RI reconstruction)")
print("-" * 70)

test_kpts = [0, 24, 47]

for ik in test_kpts:
    kpt = kernel.kpts[ik]
    print(f"\nk-point {ik}: ({kpt[0]:.4f}, {kpt[1]:.4f}, {kpt[2]:.4f})")

    # Original H(k) from file
    Hk_orig = matH.r2k(kpt).toarray()
    Sk = matS.r2k(kpt).toarray()

    # Band-RI H_loc
    H_loc_ri = result['H_loc_final'][ik]

    # Matrix norms
    norm_orig = np.linalg.norm(Hk_orig, 'fro')
    norm_ri = np.linalg.norm(H_loc_ri, 'fro')

    # Difference
    diff = Hk_orig - H_loc_ri
    diff_norm = np.linalg.norm(diff, 'fro')
    rel_diff = diff_norm / norm_orig

    print(f"  ||H_orig||_F     = {norm_orig:.6e} Ha")
    print(f"  ||H_loc_RI||_F   = {norm_ri:.6e} Ha")
    print(f"  ||H_orig - H_loc_RI||_F = {diff_norm:.6e} Ha")
    print(f"  Relative diff    = {rel_diff:.4f} ({rel_diff*100:.1f}%)")

    # Element-wise statistics
    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))
    print(f"  Max element diff = {max_abs_diff:.6e} Ha ({max_abs_diff*hartree2ev:.4f} eV)")
    print(f"  Mean element diff= {mean_abs_diff:.6e} Ha ({mean_abs_diff*hartree2ev:.4f} eV)")

    # Diagonal comparison
    diag_orig = np.diag(Hk_orig).real
    diag_ri = np.diag(H_loc_ri).real
    diag_diff = np.abs(diag_orig - diag_ri)
    print(f"  Diagonal: max diff = {np.max(diag_diff):.6e} Ha, mean = {np.mean(diag_diff):.6e} Ha")

    # Check Hermiticity
    herm_error_orig = np.linalg.norm(Hk_orig - Hk_orig.conj().T, 'fro')
    herm_error_ri = np.linalg.norm(H_loc_ri - H_loc_ri.conj().T, 'fro')
    print(f"  Hermiticity: H_orig={herm_error_orig:.2e}, H_loc_RI={herm_error_ri:.2e}")

# Step 5: Summary statistics across all k-points
print("\n" + "=" * 70)
print("[5] Summary: Matrix comparison across all k-points")
print("=" * 70)

rel_diffs_all = []
max_elem_diffs_all = []

for ik in range(kernel.nk):
    kpt = kernel.kpts[ik]
    Hk_orig = matH.r2k(kpt).toarray()
    H_loc_ri = result['H_loc_final'][ik]

    diff = Hk_orig - H_loc_ri
    rel_diff = np.linalg.norm(diff, 'fro') / np.linalg.norm(Hk_orig, 'fro')
    max_elem = np.max(np.abs(diff))

    rel_diffs_all.append(rel_diff)
    max_elem_diffs_all.append(max_elem)

rel_diffs_all = np.array(rel_diffs_all)
max_elem_diffs_all = np.array(max_elem_diffs_all)

print(f"\nRelative Frobenius norm ||H_orig - H_loc_RI|| / ||H_orig||:")
print(f"  Min:  {np.min(rel_diffs_all):.4f} ({np.min(rel_diffs_all)*100:.1f}%)")
print(f"  Mean: {np.mean(rel_diffs_all):.4f} ({np.mean(rel_diffs_all)*100:.1f}%)")
print(f"  Max:  {np.max(rel_diffs_all):.4f} ({np.max(rel_diffs_all)*100:.1f}%)")

print(f"\nMax element-wise difference (Ha):")
print(f"  Min:  {np.min(max_elem_diffs_all):.6e} ({np.min(max_elem_diffs_all)*hartree2ev:.4f} eV)")
print(f"  Mean: {np.mean(max_elem_diffs_all):.6e} ({np.mean(max_elem_diffs_all)*hartree2ev:.4f} eV)")
print(f"  Max:  {np.max(max_elem_diffs_all):.6e} ({np.max(max_elem_diffs_all)*hartree2ev:.4f} eV)")

# Step 6: Eigenvalue comparison
print("\n" + "=" * 70)
print("[6] Eigenvalue Comparison")
print("=" * 70)
print("\nCompare eigenvalues from diagonalizing H_orig vs H_loc_RI")

from scipy.linalg import eigh

eig_diffs_all = []

for ik in test_kpts:
    kpt = kernel.kpts[ik]
    print(f"\nk-point {ik}: ({kpt[0]:.4f}, {kpt[1]:.4f}, {kpt[2]:.4f})")

    Hk_orig = matH.r2k(kpt).toarray()
    Sk = matS.r2k(kpt).toarray()
    H_loc_ri = result['H_loc_final'][ik]

    # Original eigenvalues (from diagonalization we already did)
    eigs_diag = kernel.eigs[ik, :nbnd]

    # Eigenvalues from H_loc_RI (standard eigenvalue problem since H_loc is in orthonormal basis)
    eigs_ri_all = np.linalg.eigvalsh(H_loc_ri)
    # Take the nbnd largest magnitude eigenvalues (band-RI has rank=nbnd)
    eigs_ri_sorted = np.sort(eigs_ri_all[np.argsort(np.abs(eigs_ri_all))[-nbnd:]])

    eigs_diag_sorted = np.sort(eigs_diag)

    print(f"  First 5 eigenvalues (eV):")
    print(f"    From diag:    {eigs_diag_sorted[:5] * hartree2ev}")
    print(f"    From H_loc_RI:{eigs_ri_sorted[:5] * hartree2ev}")

    eig_diff = np.abs(eigs_ri_sorted - eigs_diag_sorted)
    print(f"  Max eig diff:  {np.max(eig_diff):.6e} Ha ({np.max(eig_diff)*hartree2ev:.4f} eV)")
    print(f"  Mean eig diff: {np.mean(eig_diff):.6e} Ha ({np.mean(eig_diff)*hartree2ev:.4f} eV)")

# Step 7: Convergence trend analysis
print("\n" + "=" * 70)
print("[7] Band-RI Convergence Trend")
print("=" * 70)
print("\nFrobenius norm change ||H_loc(N) - H_loc(N-1)|| / ||H_loc(N)||")
print("As N increases, this should decrease, indicating convergence.\n")

mean_errors = np.mean(result['errors'], axis=0)
for i, nbnd_w in enumerate(result['nband_list']):
    err = mean_errors[i]
    if np.isinf(err):
        print(f"  nbnd = {nbnd_w:3d}: (first window, no previous)")
    else:
        print(f"  nbnd = {nbnd_w:3d}: {err:.6e}")

# Step 8: Summary
print("\n" + "=" * 70)
print("[8] Summary")
print("=" * 70)
print(f"""
Matrix comparison results:
- H_orig(k): Original Hamiltonian from PW→AO projection (hamiltonians.h5)
- H_loc_RI(k): Band-RI reconstruction A† @ diag(ε) @ A

Key observations:
1. The matrices differ because band-RI uses only {nbnd} bands out of {nao} AOs
2. Band-RI is a PROJECTION onto the computed band subspace
3. The relative difference ({np.mean(rel_diffs_all)*100:.1f}% average) reflects
   the incompleteness of the band basis

Convergence testing:
- Band windows tested: {result['nband_list']}
- Mean Frobenius change at final window: {mean_errors[-1]:.4e}

Usage:
    kernel.diag(nbnd=N)
    result = kernel.band_ri_convergence(nband_list=[N//4, N//2, 3*N//4, N])
""")

print("Done!")
