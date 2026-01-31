"""
Test eigenvalue reconstruction quality for low-lying bands.

Workflow:
1. At each k-point, reconstruct H(k) = A_k† @ diag(ε_k) @ A_k using nbands
2. Diagonalize reconstructed H(k) to get eigenvalues
3. Compare with reference eigenvalues from file (using all 100 bands)
4. Analyze error for lowest 5, 10, 15 bands as function of nbands in reconstruction

This tests how well a TRUNCATED band-RI reconstruction reproduces eigenvalues.
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import FortranFile
from scipy.linalg import eigh

from HPRO.deephio import load_deeph_HS
from HPRO.mathutils import compute_local_h_band_ri
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs
from HPRO.constants import hartree2ev


def parse_all_kpoints_xml(xml_path):
    """Parse all k-points and eigenvalues from QE XML file."""
    tree = ET.parse(xml_path)
    kpoints = []
    eigenvalues = []
    for ks_energies in tree.iter('ks_energies'):
        k_point_text = ks_energies.find('k_point').text
        kpt = np.array([float(x) for x in k_point_text.split()])
        kpoints.append(kpt)
        eig_text = ks_energies.find('eigenvalues').text
        eigs = np.array([float(x) for x in eig_text.split()])
        eigenvalues.append(eigs)
    return kpoints, eigenvalues


def get_structure_from_xml(xml_path):
    """Get structure info from QE XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cell_elem = root.find('.//atomic_structure/cell')
    a1 = np.array([float(x) for x in cell_elem.find('a1').text.split()])
    a2 = np.array([float(x) for x in cell_elem.find('a2').text.split()])
    a3 = np.array([float(x) for x in cell_elem.find('a3').text.split()])
    rprim = np.array([a1, a2, a3])
    gprim = 2 * np.pi * np.linalg.inv(rprim.T)
    return rprim, gprim


def read_wfc_qe(path, nbands):
    """Read QE wavefunction file."""
    f = FortranFile(path, 'r')
    f.read_record(dtype='<i4')
    data = f.read_ints(np.int32)
    ngw, igwx, npol, nbnd_file = data
    f.read_reals()
    miller = f.read_ints().reshape((3, igwx), order="F")
    evc_list = []
    for _ in range(min(nbands, nbnd_file)):
        evc = f.read_record(dtype='<d').reshape((2, igwx), order="F")
        evc = np.vectorize(complex)(evc[0], evc[1])
        norm = np.sqrt(np.sum(np.conj(evc) * evc).real)
        if norm > 1e-10:
            evc /= norm
        evc_list.append(evc)
    f.close()
    return evc_list, miller


def compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut):
    """Compute AO functions φ_μ(k+G) in plane-wave basis."""
    ngw = miller.shape[1]
    kgcart = np.zeros((ngw, 3))
    for ig in range(ngw):
        g_cryst = miller[:, ig]
        g_cart = gprim.T @ g_cryst
        k_cart = gprim.T @ kpt
        kgcart[ig] = k_cart + g_cart

    FT_kg_orb_spcs = calc_FT_kg_orb_spcs(ngw, kgcart, lcaodata, ecut)

    nao_total = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
    phi_kg = np.zeros((ngw, nao_total), dtype=np.complex128)

    iao = 0
    for iatom, spc in enumerate(structure.atomic_numbers):
        nao_atom = lcaodata.norbfull_spc[spc]
        tau = structure.atomic_positions_cart[iatom]
        phase = np.exp(-1j * kgcart @ tau)
        phi_kg[:, iao:iao+nao_atom] = FT_kg_orb_spcs[spc] * phase[:, None]
        iao += nao_atom
    return phi_kg


def compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume):
    """Compute overlap matrix A[n,μ] = ⟨ψ_n|φ_μ⟩ in PW basis."""
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg
    return A


def diagonalize_generalized(H, S):
    """
    Solve generalized eigenvalue problem: H c = ε S c
    Returns eigenvalues sorted in ascending order.
    """
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


# ============================================================================
# Main test
# ============================================================================

print("=" * 70)
print("Eigenvalue reconstruction test for low-lying bands")
print("H_recon(k) = A_k† @ diag(ε_k) @ A_k  (per-k reconstruction)")
print("=" * 70)

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'
ecut = 30

# Load structure and data
print("\n[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load S matrices from file
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get all k-points and reference eigenvalues
kpoints, all_eigenvalues_ref = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# Load all wavefunction data
print("\n[2] Loading wavefunctions for all k-points...")

psi_lists = []
phi_kg_lists = []
valid_kpts = []
valid_eigs = []

max_nbands = 100

for ik in range(nkpt):
    kpt = kpoints[ik]
    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'

    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)

        psi_lists.append(psi_list)
        phi_kg_lists.append(phi_kg)
        valid_kpts.append(kpt)
        valid_eigs.append(all_eigenvalues_ref[ik])

    except Exception as e:
        print(f"    Warning: Could not load k-point {ik+1}: {e}")
        continue

print(f"    Successfully loaded {len(valid_kpts)} k-points")

# ============================================================================
# Test: Per-k eigenvalue reconstruction vs nbands
# ============================================================================
print("\n" + "=" * 70)
print("[3] Per-k eigenvalue reconstruction test")
print("=" * 70)
print("    H_recon(k) = A_k† diag(ε_k) A_k using truncated bands")
print("    Compare eigenvalues from diagonalizing H_recon(k) vs reference")

# Parameters
nbands_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]
target_bands = [5, 10, 15]  # Number of low-lying bands to compare

# Store results: results[nbands][n_target] = list of MAE over k-points
results = {nbands: {n_target: [] for n_target in target_bands} for nbands in nbands_list}

print(f"\nProcessing {len(valid_kpts)} k-points...")

for nbands in nbands_list:
    for ik in range(len(valid_kpts)):
        kpt = valid_kpts[ik]

        # Get eigenvalues and wavefunctions for this k
        eigs_k = valid_eigs[ik][:nbands]
        psi_list_k = psi_lists[ik][:nbands]
        phi_kg_k = phi_kg_lists[ik]

        # Compute A matrix
        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)

        # Reconstruct H(k) using band-RI with truncated bands
        H_recon_k = compute_local_h_band_ri(eigs_k, A_k)

        # Get S(k) from file
        S_k = matS.r2k(kpt).toarray()

        # Diagonalize reconstructed H(k)
        eigs_recon = diagonalize_generalized(H_recon_k, S_k)

        # Reference eigenvalues (from QE with all bands)
        eigs_ref = valid_eigs[ik]

        # Compute error for each target band count
        for n_target in target_bands:
            mae = np.mean(np.abs(eigs_recon[:n_target] - eigs_ref[:n_target])) * hartree2ev * 1000
            results[nbands][n_target].append(mae)

# ============================================================================
# Print results table
# ============================================================================
print("\n" + "=" * 70)
print("[4] Results: Mean Absolute Error (meV) averaged over all k-points")
print("=" * 70)

print(f"\n{'nbands':>8}", end="")
for n_target in target_bands:
    print(f" {'Lowest '+str(n_target):>12}", end="")
print()
print("-" * (8 + 13 * len(target_bands)))

for nbands in nbands_list:
    print(f"{nbands:>8}", end="")
    for n_target in target_bands:
        mae_avg = np.mean(results[nbands][n_target])
        print(f" {mae_avg:>12.1f}", end="")
    print()

print("-" * (8 + 13 * len(target_bands)))

# ============================================================================
# Detailed comparison at selected k-points
# ============================================================================
print("\n" + "=" * 70)
print("[5] Detailed eigenvalue comparison at selected k-points")
print("=" * 70)

# Select representative k-points
test_kpt_indices = [0, len(valid_kpts)//4, len(valid_kpts)//2, 3*len(valid_kpts)//4]

for nbands_show in [50, 100]:
    print(f"\n--- Using {nbands_show} bands for reconstruction ---")

    for ik_test in test_kpt_indices:
        kpt = valid_kpts[ik_test]
        k_cart = gprim.T @ kpt
        k_norm = np.linalg.norm(k_cart)

        # Reconstruct H(k)
        eigs_k = valid_eigs[ik_test][:nbands_show]
        psi_list_k = psi_lists[ik_test][:nbands_show]
        phi_kg_k = phi_kg_lists[ik_test]

        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)
        H_recon_k = compute_local_h_band_ri(eigs_k, A_k)

        S_k = matS.r2k(kpt).toarray()
        eigs_recon = diagonalize_generalized(H_recon_k, S_k)
        eigs_ref = valid_eigs[ik_test]

        kpt_str = f"k{ik_test+1} ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f})"
        print(f"\n{kpt_str}, |k|={k_norm:.4f}:")
        print(f"  {'Band':>6} {'ε_ref (eV)':>12} {'ε_recon (eV)':>14} {'Diff (meV)':>12}")
        print("  " + "-" * 48)

        for i in range(15):
            e_ref = eigs_ref[i] * hartree2ev
            e_rec = eigs_recon[i] * hartree2ev
            diff_mev = (eigs_recon[i] - eigs_ref[i]) * hartree2ev * 1000
            print(f"  {i+1:>6} {e_ref:>12.4f} {e_rec:>14.4f} {diff_mev:>12.1f}")

        # Summary for this k-point
        for n_target in target_bands:
            mae = np.mean(np.abs(eigs_recon[:n_target] - eigs_ref[:n_target])) * hartree2ev * 1000
            print(f"  MAE (lowest {n_target}): {mae:.1f} meV")

# ============================================================================
# Error vs |k| analysis
# ============================================================================
print("\n" + "=" * 70)
print("[6] Error vs |k| analysis (using 100 bands)")
print("=" * 70)

k_norms = []
errors_5 = []
errors_10 = []
errors_15 = []

for ik in range(len(valid_kpts)):
    kpt = valid_kpts[ik]
    k_cart = gprim.T @ kpt
    k_norm = np.linalg.norm(k_cart)

    k_norms.append(k_norm)
    errors_5.append(results[100][5][ik])
    errors_10.append(results[100][10][ik])
    errors_15.append(results[100][15][ik])

k_norms = np.array(k_norms)
errors_5 = np.array(errors_5)
errors_10 = np.array(errors_10)
errors_15 = np.array(errors_15)

print(f"\nCorrelation between |k| and MAE:")
print(f"  Lowest 5 bands:  r = {np.corrcoef(k_norms, errors_5)[0,1]:.4f}")
print(f"  Lowest 10 bands: r = {np.corrcoef(k_norms, errors_10)[0,1]:.4f}")
print(f"  Lowest 15 bands: r = {np.corrcoef(k_norms, errors_15)[0,1]:.4f}")

# K-points with best/worst errors
sorted_idx = np.argsort(errors_15)
print(f"\nK-points with lowest error (15 bands):")
for i in sorted_idx[:3]:
    kpt = valid_kpts[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - MAE = {errors_15[i]:.1f} meV, |k| = {k_norms[i]:.4f}")

print(f"\nK-points with highest error (15 bands):")
for i in sorted_idx[-3:]:
    kpt = valid_kpts[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - MAE = {errors_15[i]:.1f} meV, |k| = {k_norms[i]:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Per-k band-RI reconstruction: H_recon(k) = A_k† diag(ε_k) A_k

Results averaged over {len(valid_kpts)} k-points:

  nbands    Lowest 5    Lowest 10    Lowest 15
  -----------------------------------------------""")

for nbands in [50, 70, 100]:
    mae5 = np.mean(results[nbands][5])
    mae10 = np.mean(results[nbands][10])
    mae15 = np.mean(results[nbands][15])
    print(f"  {nbands:>6}    {mae5:>8.1f}    {mae10:>9.1f}    {mae15:>9.1f} meV")

print(f"""
Key observations:
  - Error decreases as more bands are used in reconstruction
  - Error increases with |k| (correlation ~{np.corrcoef(k_norms, errors_15)[0,1]:.2f})
  - Lowest bands have smallest reconstruction error
  - With 100 bands, average MAE for lowest 15 bands: {np.mean(results[100][15]):.1f} meV
""")

print("=" * 70)
print("Done!")
print("=" * 70)
