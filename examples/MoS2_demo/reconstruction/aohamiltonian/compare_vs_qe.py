"""
Compare both reconstruction methods against QE reference eigenvalues for MoS2.

1. Original reconstruction: Diagonalize H(k) from real-space reconstruction
2. Band-RI reconstruction: Diagonalize H_bandRI(k) = A† diag(ε_QE) A

Both are compared to QE reference eigenvalues.
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
    """Compute AO functions in PW basis."""
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


def diagonalize_generalized(H, S):
    """Solve generalized eigenvalue problem."""
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


# ============================================================================
# Main
# ============================================================================
print("=" * 70)
print("MoS2: Comparison of reconstruction methods vs QE reference")
print("=" * 70)

# Paths - use nscf folder which has the uniform k-grid
nscf_save_dir = '../../nscf/MoS2.save'
xml_path = f'{nscf_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'
ecut = 30
max_nbands = 100

# Load structure
print("\n[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))
print(f"    Cell volume: {cell_volume:.4f} bohr³")

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load H and S from file (original reconstruction)
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)  # Returns Hartree
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get k-points and QE eigenvalues
kpoints, all_eigenvalues = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# Load wavefunctions
print("\n[2] Loading wavefunctions...")

psi_lists = []
phi_kg_lists = []
valid_kpts = []
valid_eigs_qe = []

for ik in range(nkpt):
    kpt = kpoints[ik]
    wfc_path = f'{nscf_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)
        psi_lists.append(psi_list)
        phi_kg_lists.append(phi_kg)
        valid_kpts.append(kpt)
        valid_eigs_qe.append(all_eigenvalues[ik])
    except Exception as e:
        continue

print(f"    Successfully loaded {len(valid_kpts)} k-points")

# ============================================================================
# Compute eigenvalues from both methods and compare to QE
# ============================================================================
print("\n[3] Computing eigenvalues and comparing to QE reference...")

target_bands = [4, 8, 16]
mae_original = {n: [] for n in target_bands}
mae_bandri = {n: [] for n in target_bands}

for ik in range(len(valid_kpts)):
    kpt = valid_kpts[ik]
    eigs_qe = valid_eigs_qe[ik]  # QE reference (Hartree)

    # Original reconstruction
    Hk_orig = matH.r2k(kpt).toarray()  # Hartree
    Sk = matS.r2k(kpt).toarray()
    eigs_orig = diagonalize_generalized(Hk_orig, Sk)  # Hartree

    # Band-RI reconstruction using QE eigenvalues
    psi_list_k = psi_lists[ik][:max_nbands]
    phi_kg_k = phi_kg_lists[ik]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    nbnd = len(psi_list_k)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list_k):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg_k

    # H_bandRI uses QE eigenvalues (Hartree)
    H_bandri = compute_local_h_band_ri(eigs_qe[:nbnd], A)  # Hartree
    eigs_bandri = diagonalize_generalized(H_bandri, Sk)  # Hartree

    # Compare to QE reference
    for n in target_bands:
        mae_o = np.mean(np.abs(eigs_orig[:n] - eigs_qe[:n])) * hartree2ev * 1000
        mae_b = np.mean(np.abs(eigs_bandri[:n] - eigs_qe[:n])) * hartree2ev * 1000
        mae_original[n].append(mae_o)
        mae_bandri[n].append(mae_b)

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 70)
print("[4] Eigenvalue MAE vs QE reference (averaged over k-points)")
print("=" * 70)

print(f"\n{'Target bands':>14} {'Original (meV)':>16} {'Band-RI (meV)':>16}")
print("-" * 50)

for n in target_bands:
    avg_orig = np.mean(mae_original[n])
    avg_bandri = np.mean(mae_bandri[n])
    print(f"{n:14d} {avg_orig:16.1f} {avg_bandri:16.1f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
MoS2 reconstruction comparison vs QE reference:

Eigenvalue MAE (averaged over {len(valid_kpts)} k-points):
  Lowest 4 bands:  Original = {np.mean(mae_original[4]):.1f} meV, Band-RI = {np.mean(mae_bandri[4]):.1f} meV
  Lowest 8 bands:  Original = {np.mean(mae_original[8]):.1f} meV, Band-RI = {np.mean(mae_bandri[8]):.1f} meV
  Lowest 16 bands: Original = {np.mean(mae_original[16]):.1f} meV, Band-RI = {np.mean(mae_bandri[16]):.1f} meV

Note: Original = real-space reconstruction (T + V_loc + V_nl)
      Band-RI = A† diag(ε_QE) A approximation
""")

print("=" * 70)
print("Done!")
print("=" * 70)
