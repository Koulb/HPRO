"""
Test band-RI reconstruction for multiple k-points.
Computes A[n,μ] = ⟨ψ_n|φ_μ⟩ directly from PW wavefunctions and verifies
H_loc(k) = A_k† @ diag(ε_k) @ A_k matches the original Hamiltonian.
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import FortranFile

from HPRO.deephio import load_deeph_HS
from HPRO.constants import hartree2ev, bohr2ang
from HPRO.mathutils import compute_local_h_band_ri
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs


def read_wfc_qe(path, nbands):
    """Read QE wavefunction file."""
    f = FortranFile(path, 'r')
    rec1_raw = f.read_record(dtype='<i4')
    data = f.read_ints(np.int32)
    ngw, igwx, npol, nbnd_file = data
    data = f.read_reals()
    b1, b2, b3 = data[0:3], data[3:6], data[6:9]
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
    return evc_list, miller, np.array([b1, b2, b3])


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


# ============================================================================
# Main multi-k-point test
# ============================================================================

print("=" * 70)
print("Band-RI reconstruction test for multiple k-points")
print("=" * 70)

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'

# Parameters
nbnd = 100
ecut = 30

# Load structure
print("\n[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)

# Load H and S matrices
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get all k-points from XML
kpoints, all_eigenvalues = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")
print(f"    Number of AOs: {nao}")
print(f"    Number of bands: {nbnd}")

# Test for each k-point
print("\n[2] Testing band-RI reconstruction for each k-point...")
print("=" * 70)
print(f"{'ik':>4} {'kpt':^30} {'||H_orig||':>12} {'||H_pw||':>12} {'Error':>10}")
print("=" * 70)

errors = []
for ik in range(nkpt):
    kpt = kpoints[ik]
    eigs_pw = all_eigenvalues[ik][:nbnd]

    # Read wavefunction
    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller, b_vecs = read_wfc_qe(wfc_path, nbnd)
    except Exception as e:
        print(f"{ik+1:4d} {'ERROR: ' + str(e)[:50]}")
        continue

    # Compute AO FT at this k-point
    phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)

    # Compute overlap A = ⟨ψ|φ⟩
    A_pw = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)

    # Compute H_loc from band-RI
    H_loc_pw = compute_local_h_band_ri(eigs_pw, A_pw)

    # Get original H(k) from file
    Hk_orig = matH.r2k(kpt).toarray()

    # Compute error
    H_orig_norm = np.linalg.norm(Hk_orig, 'fro')
    H_pw_norm = np.linalg.norm(H_loc_pw, 'fro')
    error = np.linalg.norm(H_loc_pw - Hk_orig, 'fro') / H_orig_norm
    errors.append(error)

    kpt_str = f"({kpt[0]:6.3f}, {kpt[1]:6.3f}, {kpt[2]:6.3f})"
    print(f"{ik+1:4d} {kpt_str:^30} {H_orig_norm:12.4f} {H_pw_norm:12.4f} {error*100:9.2f}%")

print("=" * 70)

# Summary statistics
errors = np.array(errors)
print(f"\n[3] Summary")
print("=" * 70)
print(f"  Number of k-points tested: {len(errors)}")
print(f"  Mean error:   {np.mean(errors)*100:.2f}%")
print(f"  Std error:    {np.std(errors)*100:.2f}%")
print(f"  Min error:    {np.min(errors)*100:.2f}%")
print(f"  Max error:    {np.max(errors)*100:.2f}%")
print(f"  Median error: {np.median(errors)*100:.2f}%")

# Convergence with bands for a few k-points
print("\n[4] Band convergence at selected k-points")
print("=" * 70)

test_kpts = [0, nkpt//4, nkpt//2, 3*nkpt//4]  # Γ and a few others
for ik in test_kpts:
    kpt = kpoints[ik]
    eigs_pw = all_eigenvalues[ik]

    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller, b_vecs = read_wfc_qe(wfc_path, 100)
    except:
        continue

    phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)
    Hk_orig = matH.r2k(kpt).toarray()
    H_orig_norm = np.linalg.norm(Hk_orig, 'fro')

    kpt_str = f"k{ik+1} ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f})"
    print(f"\n  {kpt_str}:")
    for n in [20, 40, 60, 80, 100]:
        if n <= len(psi_list):
            A_n = compute_overlap_matrix_pw(psi_list[:n], phi_kg, cell_volume)
            H_n = compute_local_h_band_ri(eigs_pw[:n], A_n)
            err_n = np.linalg.norm(H_n - Hk_orig, 'fro') / H_orig_norm
            print(f"    nbnd={n:3d}: error = {err_n*100:.2f}%")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
