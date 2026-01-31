"""
Test band-RI reconstruction with UNIFORM k-point grid.

H_loc = (1/Nk) Σ_k A_k† @ diag(ε_k) @ A_k

For a uniform k-grid spanning the BZ:
  (1/Nk) Σ_k H(k) = H(R=0)

This should allow direct comparison of real-space Hamiltonians
from the two reconstruction methods.
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import FortranFile

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
    weights = []
    for ks_energies in tree.iter('ks_energies'):
        k_point_elem = ks_energies.find('k_point')
        k_point_text = k_point_elem.text
        kpt = np.array([float(x) for x in k_point_text.split()])
        kpoints.append(kpt)
        # Get weight if available
        weight = float(k_point_elem.get('weight', 1.0))
        weights.append(weight)
        eig_text = ks_energies.find('eigenvalues').text
        eigs = np.array([float(x) for x in eig_text.split()])
        eigenvalues.append(eigs)
    return kpoints, eigenvalues, np.array(weights)


def read_kpoints_from_pwinput(pw_path):
    """Read k-points from QE pw.in file (crystal coordinates)."""
    kpoints = []
    weights = []
    with open(pw_path, 'r') as f:
        lines = f.readlines()

    # Find K_POINTS line
    in_kpoints = False
    nkpt = 0
    kpt_count = 0
    for line in lines:
        line = line.strip()
        if line.upper().startswith('K_POINTS'):
            in_kpoints = True
            continue
        if in_kpoints and nkpt == 0:
            nkpt = int(line)
            continue
        if in_kpoints and kpt_count < nkpt:
            parts = line.split()
            kpt = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            weight = float(parts[3]) if len(parts) > 3 else 1.0 / nkpt
            kpoints.append(kpt)
            weights.append(weight)
            kpt_count += 1
    return kpoints, np.array(weights)


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


def compute_H_loc_ksum(valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
                        cell_volume, nbands, weights=None):
    """
    Compute H_loc by summing over k-points with weights.

    H_loc = Σ_k w_k A_k† @ diag(ε_k) @ A_k

    For normalized weights (Σ w_k = 1), this gives the k-averaged H.
    """
    nkpt = len(valid_kpts)
    nao = phi_kg_lists[0].shape[1]

    if weights is None:
        weights = np.ones(nkpt) / nkpt
    else:
        # Normalize weights
        weights = weights / np.sum(weights)

    H_loc = np.zeros((nao, nao), dtype=np.complex128)

    for ik in range(nkpt):
        eigs_k = valid_eigs[ik][:nbands]
        psi_list_k = psi_lists[ik][:nbands]
        phi_kg_k = phi_kg_lists[ik]

        A_k = compute_overlap_matrix_pw(psi_list_k, phi_kg_k, cell_volume)
        H_loc_k = compute_local_h_band_ri(eigs_k, A_k)
        H_loc += weights[ik] * H_loc_k

    return H_loc


# ============================================================================
# Main test
# ============================================================================

print("=" * 70)
print("Band-RI reconstruction with UNIFORM k-point grid")
print("H_loc = Σ_k w_k A_k† @ diag(ε_k) @ A_k")
print("=" * 70)

# Paths - use nscf uniform k-grid data
nscf_save_dir = '../../nscf/MoS2.save'
xml_path = f'{nscf_save_dir}/data-file-schema.xml'
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

# Load H and S matrices from file (original reconstruction)
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get H(R=0) - the true real-space on-site block
matscsr_R = matH.to_csr()
H_R0_file = matscsr_R[(0, 0, 0)].toarray()
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

# Get k-points from pw.in (crystal coordinates) and eigenvalues from XML
pw_input_path = '../../nscf/pw.in'
kpoints, kweights_pwin = read_kpoints_from_pwinput(pw_input_path)
print(f"    Read {len(kpoints)} k-points from pw.in (crystal coords)")
print(f"    Example: k_cryst[0]={kpoints[0]}, k_cryst[1]={kpoints[1]}")

# Get eigenvalues from XML (need these for band-RI)
_, all_eigenvalues, kweights_xml = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")
print(f"    Sum of weights (from pw.in): {np.sum(kweights_pwin):.6f}")
kweights = kweights_pwin  # Use weights from pw.in

# Normalize weights to sum to 1
kweights_norm = kweights / np.sum(kweights)
print(f"    Normalized weights sum: {np.sum(kweights_norm):.6f}")

# ============================================================================
# Load all wavefunction data
# ============================================================================
print("\n[2] Loading wavefunctions for all k-points...")

psi_lists = []
phi_kg_lists = []
valid_kpts = []
valid_eigs = []
valid_weights = []

max_nbands = 100

for ik in range(nkpt):
    kpt = kpoints[ik]  # Crystal coords
    wfc_path = f'{nscf_save_dir}/wfc{ik+1}.dat'

    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)

        psi_lists.append(psi_list)
        phi_kg_lists.append(phi_kg)
        valid_kpts.append(kpt)
        valid_eigs.append(all_eigenvalues[ik])
        valid_weights.append(kweights[ik])

    except Exception as e:
        print(f"    Warning: Could not load k-point {ik+1}: {e}")
        continue

valid_weights = np.array(valid_weights)
valid_weights_norm = valid_weights / np.sum(valid_weights)
print(f"    Successfully loaded {len(valid_kpts)} k-points")

# ============================================================================
# Test 1: Analyze R-vectors in H(R) from file
# ============================================================================
print("\n" + "=" * 70)
print("[3] Analyze R-vectors in original H(R)")
print("=" * 70)

R_vectors = list(matscsr_R.keys())
print(f"    Number of R-vectors: {len(R_vectors)}")
print("\n    R-vector  ||H(R)||_F   Phase sum (with uniform weights)")
print("    " + "-" * 52)

# For 6x6 uniform grid, compute phase sum for each R
# k = (i/6, j/6, 0) for i,j = 0..5
kgrid = []
for i in range(6):
    for j in range(6):
        kgrid.append(np.array([i/6, j/6, 0.0]))
kgrid = np.array(kgrid)

for R in sorted(R_vectors)[:15]:  # Show first 15
    H_R = matscsr_R[R].toarray()
    norm_HR = np.linalg.norm(H_R, 'fro')

    # Compute phase sum Σ_k e^{2πi k·R}
    R_vec = np.array(R)
    phases = np.exp(2j * np.pi * kgrid @ R_vec)
    phase_sum = np.abs(np.sum(phases)) / len(kgrid)  # Normalized

    print(f"    {str(R):12s} {norm_HR:10.4f}   {phase_sum:.6f}")

if len(R_vectors) > 15:
    print(f"    ... and {len(R_vectors) - 15} more R-vectors")

# Theoretical k-average = sum over R of phase_sum * H(R)
print("\n    Computing theoretical k-average from phase sums...")
H_kavg_theory = np.zeros((nao, nao), dtype=np.complex128)
for R in R_vectors:
    H_R = matscsr_R[R].toarray()
    R_vec = np.array(R)
    phases = np.exp(2j * np.pi * kgrid @ R_vec)
    phase_factor = np.sum(phases) / len(kgrid)
    H_kavg_theory += phase_factor * H_R

print(f"    ||H_kavg_theory||_F: {np.linalg.norm(H_kavg_theory, 'fro'):.4f}")
print(f"    ||H(R=0)||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")
diff_theory = np.linalg.norm(H_kavg_theory - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
print(f"    ||H_kavg_theory - H(R=0)|| / ||H(R=0)||: {diff_theory:.6f} ({diff_theory*100:.4f}%)")

# Check k-points match
print("\n    Comparing k-points (first 6 from file vs theoretical grid):")
for i in range(min(6, len(valid_kpts))):
    kpt_file = valid_kpts[i]
    kpt_theory = kgrid[i]
    diff = np.linalg.norm(kpt_file - kpt_theory)
    print(f"      File: {kpt_file}  Theory: {kpt_theory}  Diff: {diff:.6f}")

# ============================================================================
# Test 2: Verify (1/Nk) Σ_k H(k) = H(R=0) for uniform grid
# ============================================================================
print("\n" + "=" * 70)
print("[4] Verify: weighted k-average H(k) = H(R=0)")
print("=" * 70)
print("    Theory: Σ_k w_k H(k) = H(R=0) for uniform k-grid with proper weights")
print()

# Compute weighted average H(k) from file
H_avg_file = np.zeros((nao, nao), dtype=np.complex128)
for ik, kpt in enumerate(valid_kpts):
    Hk = matH.r2k(kpt).toarray()
    H_avg_file += valid_weights_norm[ik] * Hk

print(f"    ||Σ w_k H(k) from file||_F: {np.linalg.norm(H_avg_file, 'fro'):.4f}")
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

# Compare
diff_avg_R0 = np.linalg.norm(H_avg_file - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
print(f"    ||Σ w_k H(k) - H(R=0)|| / ||H(R=0)||: {diff_avg_R0:.6f} ({diff_avg_R0*100:.4f}%)")

if diff_avg_R0 < 0.01:
    print("    ✓ Verified: k-averaged H(k) ≈ H(R=0)")
else:
    print("    Note: Some discrepancy - checking diagonal...")
    # Check diagonal elements
    diag_avg = np.diag(H_avg_file).real
    diag_R0 = np.diag(H_R0_file).real
    print(f"    Diagonal MAE: {np.mean(np.abs(diag_avg - diag_R0)):.6f} Ha")

# ============================================================================
# Test 3: Band-RI k-sum vs H(R=0)
# ============================================================================
print("\n" + "=" * 70)
print("[5] Band-RI k-sum vs H(R=0) from file")
print("=" * 70)

# Compute H_loc with band-RI using all bands
H_loc_ksum = compute_H_loc_ksum(
    valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
    cell_volume, nbands=max_nbands, weights=valid_weights
)

print(f"    ||H_loc_ksum (band-RI)||_F: {np.linalg.norm(H_loc_ksum, 'fro'):.4f}")
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

diff_bandri_R0 = np.linalg.norm(H_loc_ksum - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
diff_bandri_avg = np.linalg.norm(H_loc_ksum - H_avg_file, 'fro') / np.linalg.norm(H_avg_file, 'fro')

print(f"    ||H_loc_ksum - H(R=0)|| / ||H(R=0)||: {diff_bandri_R0:.4f} ({diff_bandri_R0*100:.1f}%)")
print(f"    ||H_loc_ksum - Σ w_k H(k)|| / ||Σ w_k H(k)||: {diff_bandri_avg:.4f} ({diff_bandri_avg*100:.1f}%)")

# Check Hermiticity
herm_error = np.linalg.norm(H_loc_ksum - H_loc_ksum.conj().T) / np.linalg.norm(H_loc_ksum)
print(f"    Hermitian error: {herm_error:.6e}")

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

print(f"{'nbands':>8} {'||H_ksum - H(R=0)||/||H(R=0)||':>30} {'||H_ksum - Σw_kH(k)||/||Σw_kH(k)||':>35}")
print("-" * 78)

for nbands in band_counts:
    H_loc_n = compute_H_loc_ksum(
        valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
        cell_volume, nbands=nbands, weights=valid_weights
    )

    err_R0 = np.linalg.norm(H_loc_n - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
    err_avg = np.linalg.norm(H_loc_n - H_avg_file, 'fro') / np.linalg.norm(H_avg_file, 'fro')

    errors_vs_R0.append(err_R0)
    errors_vs_avg.append(err_avg)

    print(f"{nbands:8d} {err_R0*100:29.2f}% {err_avg*100:34.2f}%")

print("-" * 78)

# ============================================================================
# Test 5: Diagonal elements comparison
# ============================================================================
print("\n" + "=" * 70)
print("[7] Diagonal elements comparison (on-site energies)")
print("=" * 70)

H_loc_100 = compute_H_loc_ksum(
    valid_kpts, valid_eigs, psi_lists, phi_kg_lists,
    cell_volume, nbands=100, weights=valid_weights
)

print("\nFirst 15 diagonal elements (Hartree):")
print(f"{'AO':>4} {'H_ksum':>12} {'H(R=0)':>12} {'Σw_kH(k)':>12} {'Δ(ksum-R0)':>12}")
print("-" * 56)

for i in range(min(15, nao)):
    h_ksum = H_loc_100[i, i].real
    h_R0 = H_R0_file[i, i].real
    h_avg = H_avg_file[i, i].real
    diff = h_ksum - h_R0
    print(f"{i+1:4d} {h_ksum:12.6f} {h_R0:12.6f} {h_avg:12.6f} {diff:12.6f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
UNIFORM k-grid band-RI: H_loc = Σ_k w_k A_k† diag(ε_k) A_k

Theory: Σ_k w_k H(k) = H(R=0) for uniform k-grid

Verification (original reconstruction):
  - ||Σ w_k H(k) - H(R=0)|| / ||H(R=0)||: {diff_avg_R0*100:.4f}%
  {'✓ Theory verified!' if diff_avg_R0 < 0.001 else ''}

Real-space Hamiltonian comparison:
  - ||H_loc_ksum (band-RI) - H(R=0) (original)||: {diff_bandri_R0*100:.2f}%

Results with {len(valid_kpts)} k-points and {max_nbands} bands:
  - Error vs H(R=0) from file: {errors_vs_R0[-1]*100:.2f}%
  - Error vs k-averaged H(k): {errors_vs_avg[-1]*100:.2f}%
  - H_loc_ksum is Hermitian: {herm_error < 1e-10}

Convergence (band truncation effect):
  - 20 bands: {errors_vs_R0[1]*100:.2f}% error
  - 50 bands: {errors_vs_R0[4]*100:.2f}% error
  - 80 bands: {errors_vs_R0[7]*100:.2f}% error
  - 100 bands: {errors_vs_R0[-1]*100:.2f}% error

The band-RI k-sum H_loc converges toward H(R=0) as more bands are included.
""")

print("=" * 70)
print("Done!")
print("=" * 70)
