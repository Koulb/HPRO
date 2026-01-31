"""
Compare eigenvalues from two reconstruction methods for low-lying bands.

Two methods:
1. Original reconstruction: Diagonalize H(k) from file (PW→AO projection)
2. Band-RI reconstruction: Diagonalize H_recon(k) = A_k† diag(ε_k) A_k

Uses eigenvalues from Original method in the band-RI formula.
Compare how well band-RI reproduces Original eigenvalues as function of nbands.
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
    """Parse all k-points from QE XML file (only k-points, not eigenvalues)."""
    tree = ET.parse(xml_path)
    kpoints = []
    for ks_energies in tree.iter('ks_energies'):
        k_point_text = ks_energies.find('k_point').text
        kpt = np.array([float(x) for x in k_point_text.split()])
        kpoints.append(kpt)
    return kpoints


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
    _, igwx, _, nbnd_file = data
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
    """Solve generalized eigenvalue problem: H c = ε S c"""
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


# ============================================================================
# Main comparison
# ============================================================================

print("=" * 70)
print("Comparison of two reconstruction methods for low-lying bands")
print("=" * 70)
print("""
Methods compared:
  1. Original: Diagonalize H(k) from file (PW→AO projection)
  2. Band-RI: Diagonalize H_recon(k) = A_k† diag(ε_k) A_k

Band-RI uses eigenvalues from Original method.
""")

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'
ecut = 30

# Load structure and data
print("[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")
print(f"    Cell volume: {cell_volume:.4f} bohr³")

# Load H and S matrices from file (original reconstruction)
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get k-points from XML
kpoints = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# Load all wavefunction data
print("\n[2] Loading wavefunctions for all k-points...")

psi_lists = []
phi_kg_lists = []
valid_kpts = []
eigs_orig_lists = []  # Eigenvalues from Original method

max_nbands = 100

for ik in range(nkpt):
    kpt = kpoints[ik]
    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'

    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)

        # Get eigenvalues from Original method (diagonalize H(k) from file)
        H_k = matH.r2k(kpt).toarray()
        S_k = matS.r2k(kpt).toarray()
        eigs_orig = diagonalize_generalized(H_k, S_k)

        psi_lists.append(psi_list)
        phi_kg_lists.append(phi_kg)
        valid_kpts.append(kpt)
        eigs_orig_lists.append(eigs_orig)

    except Exception as e:
        print(f"    Warning: Could not load k-point {ik+1}: {e}")
        continue

print(f"    Successfully loaded {len(valid_kpts)} k-points")

# ============================================================================
# Compute eigenvalues from both methods
# ============================================================================
print("\n[3] Computing eigenvalues from both methods...")

target_bands = [5, 10, 15]
nbands_list = [20, 30, 40, 50, 60, 70, 80, 90]  # Limited by nao

# Storage for results
results_diff = {nb: {n: [] for n in target_bands} for nb in nbands_list}
detailed_results = []

for ik in range(len(valid_kpts)):
    kpt = valid_kpts[ik]
    eigs_orig = eigs_orig_lists[ik]

    # Get S(k) from file
    S_k = matS.r2k(kpt).toarray()

    # Method 2: Band-RI reconstruction for each nbands
    phi_kg_k = phi_kg_lists[ik]

    for nbands in nbands_list:
        # Use eigenvalues from Original method for band-RI
        eigs_for_ri = eigs_orig[:nbands]
        psi_list_k = psi_lists[ik][:nbands]

        # Compute A matrix
        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)

        # Reconstruct H(k) using band-RI
        H_recon_k = compute_local_h_band_ri(eigs_for_ri, A_k)

        # Diagonalize reconstructed H(k)
        eigs_bandri = diagonalize_generalized(H_recon_k, S_k)

        # Compute difference (Band-RI vs Original)
        for n_target in target_bands:
            mae = np.mean(np.abs(eigs_bandri[:n_target] - eigs_orig[:n_target])) * hartree2ev * 1000
            results_diff[nbands][n_target].append(mae)

    # Store detailed results for selected k-points
    if ik in [0, len(valid_kpts)//4, len(valid_kpts)//2, 3*len(valid_kpts)//4]:
        detailed_results.append({
            'ik': ik,
            'kpt': kpt,
            'eigs_orig': eigs_orig
        })

# ============================================================================
# Results table
# ============================================================================
print("\n" + "=" * 70)
print("[4] Results: MAE (meV) between Band-RI and Original methods")
print("    (averaged over all k-points)")
print("=" * 70)

print(f"\n{'nbands':>12}", end="")
for n in target_bands:
    print(f" {'Lowest '+str(n):>12}", end="")
print()
print("-" * 50)

for nbands in nbands_list:
    print(f"{nbands:>12}", end="")
    for n in target_bands:
        mae = np.mean(results_diff[nbands][n])
        print(f" {mae:>12.1f}", end="")
    print()

print("-" * 50)

# ============================================================================
# Detailed comparison at selected k-points
# ============================================================================
print("\n" + "=" * 70)
print("[5] Detailed eigenvalue comparison at selected k-points")
print("=" * 70)

for result in detailed_results:
    ik = result['ik']
    kpt = result['kpt']
    eigs_orig = result['eigs_orig']

    k_cart = gprim.T @ kpt
    k_norm = np.linalg.norm(k_cart)

    # Compute band-RI eigenvalues for this k-point
    phi_kg_k = phi_kg_lists[ik]
    S_k = matS.r2k(kpt).toarray()

    eigs_bandri_dict = {}
    for nbands in [20, 50, 90]:
        eigs_for_ri = eigs_orig[:nbands]
        psi_list_k = psi_lists[ik][:nbands]
        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)
        H_recon_k = compute_local_h_band_ri(eigs_for_ri, A_k)
        eigs_bandri = diagonalize_generalized(H_recon_k, S_k)
        eigs_bandri_dict[nbands] = eigs_bandri

    kpt_str = f"k{ik+1} ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f})"
    print(f"\n{kpt_str}, |k|={k_norm:.4f}:")
    print(f"  {'Band':>6} {'ε_orig':>10} {'ε_RI20':>10} {'ε_RI50':>10} {'ε_RI90':>10} {'Δ20':>8} {'Δ50':>8} {'Δ90':>8}")
    print("  " + "-" * 80)

    for i in range(15):
        e_orig = eigs_orig[i] * hartree2ev
        e_ri20 = eigs_bandri_dict[20][i] * hartree2ev
        e_ri50 = eigs_bandri_dict[50][i] * hartree2ev
        e_ri90 = eigs_bandri_dict[90][i] * hartree2ev
        d20 = (eigs_bandri_dict[20][i] - eigs_orig[i]) * hartree2ev * 1000
        d50 = (eigs_bandri_dict[50][i] - eigs_orig[i]) * hartree2ev * 1000
        d90 = (eigs_bandri_dict[90][i] - eigs_orig[i]) * hartree2ev * 1000
        print(f"  {i+1:>6} {e_orig:>10.3f} {e_ri20:>10.3f} {e_ri50:>10.3f} {e_ri90:>10.3f} {d20:>8.1f} {d50:>8.1f} {d90:>8.1f}")

    # Summary MAE for this k-point
    for nbands in [20, 50, 90]:
        eigs_ri = eigs_bandri_dict[nbands]
        mae5 = np.mean(np.abs(eigs_ri[:5] - eigs_orig[:5])) * hartree2ev * 1000
        mae10 = np.mean(np.abs(eigs_ri[:10] - eigs_orig[:10])) * hartree2ev * 1000
        mae15 = np.mean(np.abs(eigs_ri[:15] - eigs_orig[:15])) * hartree2ev * 1000
        print(f"  MAE (RI{nbands}):  L5={mae5:>6.1f}  L10={mae10:>6.1f}  L15={mae15:>6.1f} meV")

# ============================================================================
# Convergence analysis
# ============================================================================
print("\n" + "=" * 70)
print("[6] Band-RI convergence analysis")
print("=" * 70)
print("\nHow does Band-RI approach Original as nbands increases?")

print(f"\n{'nbands':>8} {'Lowest 5':>12} {'Lowest 10':>12} {'Lowest 15':>12}")
print("-" * 50)

for nbands in nbands_list:
    mae5 = np.mean(results_diff[nbands][5])
    mae10 = np.mean(results_diff[nbands][10])
    mae15 = np.mean(results_diff[nbands][15])
    print(f"{nbands:>8} {mae5:>12.1f} {mae10:>12.1f} {mae15:>12.1f}")

# ============================================================================
# Error vs |k| analysis
# ============================================================================
print("\n" + "=" * 70)
print("[7] Error vs |k| analysis (using 90 bands)")
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
    errors_5.append(results_diff[90][5][ik])
    errors_10.append(results_diff[90][10][ik])
    errors_15.append(results_diff[90][15][ik])

k_norms = np.array(k_norms)
errors_5 = np.array(errors_5)
errors_10 = np.array(errors_10)
errors_15 = np.array(errors_15)

print(f"\nCorrelation between |k| and MAE:")
print(f"  Lowest 5 bands:  r = {np.corrcoef(k_norms, errors_5)[0,1]:.4f}")
print(f"  Lowest 10 bands: r = {np.corrcoef(k_norms, errors_10)[0,1]:.4f}")
print(f"  Lowest 15 bands: r = {np.corrcoef(k_norms, errors_15)[0,1]:.4f}")

# K-points with smallest/largest differences
sorted_idx = np.argsort(errors_15)
print(f"\nK-points with smallest difference (lowest 15):")
for i in sorted_idx[:3]:
    kpt = valid_kpts[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - MAE = {errors_15[i]:.1f} meV, |k| = {k_norms[i]:.4f}")

print(f"\nK-points with largest difference (lowest 15):")
for i in sorted_idx[-3:]:
    kpt = valid_kpts[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - MAE = {errors_15[i]:.1f} meV, |k| = {k_norms[i]:.4f}")

# ============================================================================
# Final summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

mae_20_5 = np.mean(results_diff[20][5])
mae_20_10 = np.mean(results_diff[20][10])
mae_20_15 = np.mean(results_diff[20][15])

mae_50_5 = np.mean(results_diff[50][5])
mae_50_10 = np.mean(results_diff[50][10])
mae_50_15 = np.mean(results_diff[50][15])

mae_90_5 = np.mean(results_diff[90][5])
mae_90_10 = np.mean(results_diff[90][10])
mae_90_15 = np.mean(results_diff[90][15])

print(f"""
Comparison: Band-RI vs Original reconstruction
(MAE in meV, averaged over {len(valid_kpts)} k-points)

Band-RI uses eigenvalues from Original method in H_recon = A† diag(ε) A

                      Lowest 5    Lowest 10    Lowest 15
  --------------------------------------------------------
  Band-RI (20 bands)  {mae_20_5:>9.1f}    {mae_20_10:>9.1f}    {mae_20_15:>9.1f}
  Band-RI (50 bands)  {mae_50_5:>9.1f}    {mae_50_10:>9.1f}    {mae_50_15:>9.1f}
  Band-RI (90 bands)  {mae_90_5:>9.1f}    {mae_90_10:>9.1f}    {mae_90_15:>9.1f}

Key observations:
  1. Band-RI converges to Original as nbands increases
  2. With 20 bands: ~{mae_20_15:.0f} meV difference for lowest 15 bands
  3. With 90 bands: ~{mae_90_15:.0f} meV difference for lowest 15 bands
  4. Error correlation with |k|: r = {np.corrcoef(k_norms, errors_15)[0,1]:.2f}
""")

print("=" * 70)
print("Done!")
print("=" * 70)
