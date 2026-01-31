"""
Test band-RI reconstruction with k-point summation.

H_loc = (1/Nk) Σ_k A_k† @ diag(ε_k) @ A_k

This sums the band-RI formula over all k-points to get the total
local Hamiltonian matrix.

For a complete basis and sufficient bands, this should converge to
the real-space Hamiltonian H(R=0).
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import FortranFile

from HPRO.deephio import load_deeph_HS
from HPRO.mathutils import compute_local_h_band_ri
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs


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
    rec1_raw = f.read_record(dtype='<i4')
    data = f.read_ints(np.int32)
    ngw, igwx, npol, nbnd_file = data
    data = f.read_reals()
    miller = f.read_ints().reshape((3, igwx), order="F")
    evc_list = []
    for iband in range(min(nbands, nbnd_file)):
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


def compute_H_loc_ksum(kpoints, all_eigenvalues, psi_lists, phi_kg_lists,
                        cell_volume, nbands, weights=None):
    """
    Compute H_loc by summing over k-points.

    H_loc = (1/Nk) Σ_k A_k† @ diag(ε_k) @ A_k

    or with weights:

    H_loc = Σ_k w_k A_k† @ diag(ε_k) @ A_k

    Args:
        kpoints: list of k-points
        all_eigenvalues: list of eigenvalue arrays for each k
        psi_lists: list of wavefunction lists for each k
        phi_kg_lists: list of phi_kg arrays for each k
        cell_volume: cell volume
        nbands: number of bands to use
        weights: k-point weights (if None, use 1/Nk)

    Returns:
        H_loc: (nao, nao) complex array
    """
    nkpt = len(kpoints)
    nao = phi_kg_lists[0].shape[1]

    if weights is None:
        weights = np.ones(nkpt) / nkpt

    H_loc = np.zeros((nao, nao), dtype=np.complex128)

    for ik in range(nkpt):
        # Get data for this k-point
        eigs_k = all_eigenvalues[ik][:nbands]
        psi_list_k = psi_lists[ik][:nbands]
        phi_kg_k = phi_kg_lists[ik]

        # Compute A matrix for this k-point
        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)

        # Compute H_loc(k) = A† diag(ε) A
        H_loc_k = compute_local_h_band_ri(eigs_k, A_k)

        # Add to total with weight
        H_loc += weights[ik] * H_loc_k

    return H_loc


# ============================================================================
# Main test
# ============================================================================

print("=" * 70)
print("Band-RI reconstruction with k-point summation")
print("H_loc = (1/Nk) Σ_k A_k† @ diag(ε_k) @ A_k")
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
print(f"    Cell volume: {cell_volume:.4f} bohr³")

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load H and S matrices from file
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get H(k=Γ) - this is the Fourier transform at k=0, NOT H(R=0)
H_k_gamma = matH.r2k(np.array([0., 0., 0.])).toarray()
print(f"    ||H(k=Γ)||_F: {np.linalg.norm(H_k_gamma, 'fro'):.4f}")

# Get true H(R=0) - the real-space on-site block
# This is what k-sum should converge to: (1/Nk) Σ_k H(k) = H(R=0)
matscsr_R = matH.to_csr()
H_R0_file = matscsr_R[(0, 0, 0)].toarray()
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

# Get all k-points
kpoints, all_eigenvalues = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# ============================================================================
# Load all wavefunction data
# ============================================================================
print("\n[2] Loading wavefunctions for all k-points...")

psi_lists = []
phi_kg_lists = []
valid_kpts = []
valid_eigs = []

max_nbands = 100  # Maximum bands available

for ik in range(nkpt):
    kpt = kpoints[ik]
    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'

    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)

        psi_lists.append(psi_list)
        phi_kg_lists.append(phi_kg)
        valid_kpts.append(kpt)
        valid_eigs.append(all_eigenvalues[ik])

    except Exception as e:
        print(f"    Warning: Could not load k-point {ik+1}: {e}")
        continue

print(f"    Successfully loaded {len(valid_kpts)} k-points")

# ============================================================================
# Test 1: Compare k-sum H_loc with H(R=0) from file
# ============================================================================
print("\n" + "=" * 70)
print("[3] Test: H_loc from k-sum vs H(R=0) from file")
print("=" * 70)
print("    Key relationship: (1/Nk) Σ_k H(k) = H(R=0)")
print()

# Compute H_loc with k-sum using all bands
H_loc_ksum = compute_H_loc_ksum(
    valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
    cell_volume, nbands=max_nbands
)

# Compare with H(R=0) from file (the true real-space on-site block)
diff_R0 = np.linalg.norm(H_loc_ksum - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')

print(f"    ||H_loc_ksum||_F: {np.linalg.norm(H_loc_ksum, 'fro'):.4f}")
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")
print(f"    ||H_loc_ksum - H(R=0)|| / ||H(R=0)||: {diff_R0:.4f} ({diff_R0*100:.1f}%)")

# Also compare with H(k=Γ) for reference
diff_gamma = np.linalg.norm(H_loc_ksum - H_k_gamma, 'fro') / np.linalg.norm(H_k_gamma, 'fro')
print(f"    ||H_loc_ksum - H(k=Γ)|| / ||H(k=Γ)||: {diff_gamma:.4f} ({diff_gamma*100:.1f}%)")

# Check if H_loc_ksum is Hermitian
herm_error = np.linalg.norm(H_loc_ksum - H_loc_ksum.conj().T) / np.linalg.norm(H_loc_ksum)
print(f"    Hermitian error: {herm_error:.6e}")

# ============================================================================
# Test 2: Verify relationship: (1/Nk) Σ_k H(k) = H(R=0)
# ============================================================================
print("\n" + "=" * 70)
print("[4] Verify: k-averaged H(k) vs H(R=0) from file")
print("=" * 70)
print("    Theory: (1/Nk) Σ_k H(k) = H(R=0) for uniform k-grid")
print()

# Compute average H(k) from file
H_avg_file = np.zeros((nao, nao), dtype=np.complex128)
for kpt in valid_kpts:
    Hk = matH.r2k(kpt).toarray()
    H_avg_file += Hk / len(valid_kpts)

print(f"    ||⟨H(k)⟩_k from file||_F: {np.linalg.norm(H_avg_file, 'fro'):.4f}")
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

# Verify H_avg ≈ H(R=0)
diff_avg_R0 = np.linalg.norm(H_avg_file - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
print(f"    ||⟨H(k)⟩_k - H(R=0)|| / ||H(R=0)||: {diff_avg_R0:.6f} ({diff_avg_R0*100:.4f}%)")

if diff_avg_R0 < 0.01:
    print("    ✓ Verified: k-averaged H(k) = H(R=0)")
else:
    print("    Note: k-grid may not be uniform or complete")

# ============================================================================
# Test 3: Compare band-RI with both references
# ============================================================================
print("\n" + "=" * 70)
print("[5] Compare band-RI H_loc with references")
print("=" * 70)

diff_ksum_avg = np.linalg.norm(H_loc_ksum - H_avg_file, 'fro') / np.linalg.norm(H_avg_file, 'fro')
diff_ksum_R0 = np.linalg.norm(H_loc_ksum - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')

print(f"    ||H_loc_ksum - ⟨H(k)⟩_k|| / ||⟨H(k)⟩||: {diff_ksum_avg:.4f} ({diff_ksum_avg*100:.1f}%)")
print(f"    ||H_loc_ksum - H(R=0)|| / ||H(R=0)||: {diff_ksum_R0:.4f} ({diff_ksum_R0*100:.1f}%)")

# ============================================================================
# Test 4: Convergence with number of bands
# ============================================================================
print("\n" + "=" * 70)
print("[6] Convergence: H_loc_ksum vs nbands")
print("=" * 70)
print("    Comparing with H(R=0) from file (real-space reference)")
print()

band_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
errors_vs_R0 = []
errors_vs_avg = []

print(f"{'nbands':>8} {'||H_ksum - H(R=0)||/||H(R=0)||':>30} {'||H_ksum - ⟨H(k)⟩||/||⟨H(k)⟩||':>32}")
print("-" * 75)

for nbands in band_counts:
    H_loc_n = compute_H_loc_ksum(
        valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
        cell_volume, nbands=nbands
    )

    err_R0 = np.linalg.norm(H_loc_n - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
    err_avg = np.linalg.norm(H_loc_n - H_avg_file, 'fro') / np.linalg.norm(H_avg_file, 'fro')

    errors_vs_R0.append(err_R0)
    errors_vs_avg.append(err_avg)

    print(f"{nbands:8d} {err_R0*100:29.2f}% {err_avg*100:31.2f}%")

print("-" * 75)

# ============================================================================
# Test 5: Diagonal elements comparison
# ============================================================================
print("\n" + "=" * 70)
print("[7] Diagonal elements comparison (on-site energies)")
print("=" * 70)

H_loc_100 = compute_H_loc_ksum(
    valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
    cell_volume, nbands=100
)

print("\nFirst 10 diagonal elements (Hartree):")
print(f"{'AO':>4} {'H_ksum':>12} {'H(R=0)':>12} {'⟨H(k)⟩':>12} {'Diff(ksum-R0)':>15}")
print("-" * 60)

for i in range(min(10, nao)):
    h_ksum = H_loc_100[i, i].real
    h_R0 = H_R0_file[i, i].real
    h_avg = H_avg_file[i, i].real
    diff = h_ksum - h_R0
    print(f"{i+1:4d} {h_ksum:12.6f} {h_R0:12.6f} {h_avg:12.6f} {diff:15.6f}")

# ============================================================================
# Test 6: Per-k-point contribution analysis
# ============================================================================
print("\n" + "=" * 70)
print("[8] Per-k-point contribution to H_loc")
print("=" * 70)

print("\nContribution from each k-point (using 100 bands):")
print(f"{'ik':>4} {'kpt':^30} {'||H_loc(k)||_F':>15}")
print("-" * 55)

H_loc_contributions = []
for ik in range(len(valid_kpts)):
    kpt = valid_kpts[ik]
    eigs_k = valid_eigs[ik][:100]
    psi_list_k = psi_lists[ik][:100]
    phi_kg_k = phi_kg_lists[ik]

    A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)
    H_loc_k = compute_local_h_band_ri(eigs_k, A_k)

    H_loc_contributions.append(H_loc_k)

    kpt_str = f"({kpt[0]:6.3f}, {kpt[1]:6.3f}, {kpt[2]:6.3f})"
    print(f"{ik+1:4d} {kpt_str:^30} {np.linalg.norm(H_loc_k, 'fro'):15.4f}")

# Check variance across k-points
H_loc_array = np.array(H_loc_contributions)
H_loc_mean = np.mean(H_loc_array, axis=0)
H_loc_std = np.std(np.abs(H_loc_array), axis=0)

print(f"\nVariance analysis:")
print(f"    Mean ||H_loc(k)||_F across k: {np.mean([np.linalg.norm(h) for h in H_loc_contributions]):.4f}")
print(f"    Std ||H_loc(k)||_F across k: {np.std([np.linalg.norm(h) for h in H_loc_contributions]):.4f}")
print(f"    Coefficient of variation: {np.std([np.linalg.norm(h) for h in H_loc_contributions]) / np.mean([np.linalg.norm(h) for h in H_loc_contributions]):.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
k-point summation band-RI: H_loc = (1/Nk) Σ_k A_k† diag(ε_k) A_k

Theory: (1/Nk) Σ_k H(k) = H(R=0) for uniform k-grid

Verification:
  - ||⟨H(k)⟩_k - H(R=0)|| / ||H(R=0)||: {diff_avg_R0*100:.4f}% (should be ~0)

Results with {len(valid_kpts)} k-points and {max_nbands} bands:
  - Error vs H(R=0) from file: {errors_vs_R0[-1]*100:.2f}%
  - Error vs k-averaged H(k): {errors_vs_avg[-1]*100:.2f}%
  - H_loc_ksum is Hermitian: {herm_error < 1e-10}

Convergence with bands (vs H(R=0)):
  - 20 bands: {errors_vs_R0[1]*100:.2f}% error
  - 50 bands: {errors_vs_R0[4]*100:.2f}% error
  - 100 bands: {errors_vs_R0[-1]*100:.2f}% error

The k-summed band-RI H_loc should converge to H(R=0) from the original reconstruction.
""")

print("=" * 70)
print("Done!")
print("=" * 70)
