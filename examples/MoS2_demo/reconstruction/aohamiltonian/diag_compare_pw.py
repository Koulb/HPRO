"""
Compare band-RI Hamiltonian reconstruction using:
1. Original method: diagonalize H(k) to get eigenvectors A and eigenvalues ε
2. Direct PW method: compute A = ⟨ψ|φ⟩ directly from PW overlap

Both methods should give H_loc = A† @ diag(ε) @ A
"""

import numpy as np
import xml.etree.ElementTree as ET
import re
from scipy.io import FortranFile

from HPRO.lcaodiag import LCAODiagKernel
from HPRO.deephio import load_deeph_HS
from HPRO.constants import hartree2ev, bohr2ang
from HPRO.mathutils import compute_local_h_band_ri
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs


def calculate_braket(a, b):
    """Calculate <a|b> = sum(a* @ b)"""
    return np.sum(np.conj(a) * b)


def read_wfc_qe(path, nbands):
    """
    Read QE wavefunction file (wfc*.dat).

    Returns:
        evc_list: list of nbands arrays, each (ngw,) complex
        miller: (3, igwx) int - G-vector indices
        b1, b2, b3: reciprocal lattice vectors
        ik: k-point index
        xk: k-point in crystal coordinates
    """
    f = FortranFile(path, 'r')

    # First record: k-point info
    data = f.read_ints(np.int32)
    ik = data[0]
    # k-point is stored as int but should be read as reals
    # Re-read as reals
    f.close()

    f = FortranFile(path, 'r')
    # Read first record properly - it contains mixed types
    # Format: ik (int), xk (3 reals), ispin (int)
    rec1_raw = f.read_record(dtype='<i4')  # Read as int first

    # Second record: ngw, igwx, npol, nbnd
    data = f.read_ints(np.int32)
    ngw, igwx, npol, nbnd_file = data

    # Third record: reciprocal lattice vectors
    data = f.read_reals()
    b1, b2, b3 = data[0:3], data[3:6], data[6:9]

    # Fourth record: Miller indices
    miller = f.read_ints().reshape((3, igwx), order="F")

    # Read wavefunctions
    evc_list = []
    for iband in range(nbands):
        evc = f.read_record(dtype='<d').reshape((2, igwx), order="F")
        evc = np.vectorize(complex)(evc[0], evc[1])
        # Normalize
        norm = np.sqrt(calculate_braket(evc, evc))
        if norm > 1e-10:
            evc /= norm
        evc_list.append(evc)

    f.close()

    return evc_list, miller, np.array([b1, b2, b3])


def parse_eigenvalues_xml(xml_path, ik):
    """
    Parse eigenvalues from QE XML file for k-point ik (1-indexed).

    Returns:
        eigenvalues in Hartree
        k-point in crystal coordinates
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    count = 1
    for ks_energies in tree.iter('ks_energies'):
        if count == ik:
            k_point_text = ks_energies.find('k_point').text
            kpt = np.array([float(x) for x in k_point_text.split()])

            eig_text = ks_energies.find('eigenvalues').text
            eigenvalues = np.array([float(x) for x in eig_text.split()])
            return eigenvalues, kpt
        count += 1

    raise ValueError(f"k-point {ik} not found in XML")


def get_structure_from_xml(xml_path):
    """Get structure info from QE XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get cell
    cell_elem = root.find('.//atomic_structure/cell')
    a1 = np.array([float(x) for x in cell_elem.find('a1').text.split()])
    a2 = np.array([float(x) for x in cell_elem.find('a2').text.split()])
    a3 = np.array([float(x) for x in cell_elem.find('a3').text.split()])
    rprim = np.array([a1, a2, a3])  # In bohr

    # Get reciprocal lattice
    gprim = 2 * np.pi * np.linalg.inv(rprim.T)

    return rprim, gprim


def compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut):
    """
    Compute AO functions φ_μ(k+G) in plane-wave basis.

    Returns:
        phi_kg: (ngw, nao) complex array
    """
    # k+G vectors in Cartesian coordinates
    ngw = miller.shape[1]
    kgcart = np.zeros((ngw, 3))
    for ig in range(ngw):
        g_cryst = miller[:, ig]
        g_cart = gprim.T @ g_cryst  # G in Cartesian
        k_cart = gprim.T @ kpt  # k in Cartesian
        kgcart[ig] = k_cart + g_cart

    # Get FT of AO functions at k+G points
    FT_kg_orb_spcs = calc_FT_kg_orb_spcs(ngw, kgcart, lcaodata, ecut)

    # Build full phi_kg matrix including all atoms
    nao_total = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
    phi_kg = np.zeros((ngw, nao_total), dtype=np.complex128)

    iao = 0
    for iatom, spc in enumerate(structure.atomic_numbers):
        nao_atom = lcaodata.norbfull_spc[spc]

        # Phase factor for atom position: e^{-i(k+G)·τ}
        tau = structure.atomic_positions_cart[iatom]  # In bohr
        phase = np.exp(-1j * kgcart @ tau)

        # AO contribution from this atom
        phi_kg[:, iao:iao+nao_atom] = FT_kg_orb_spcs[spc] * phase[:, None]
        iao += nao_atom

    return phi_kg


def compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume):
    """
    Compute overlap matrix A[n,μ] = ⟨ψ_n|φ_μ⟩ in PW basis.

    QE convention: ψ(r) = (1/√Ω) Σ_G c_G e^{i(k+G)·r}
    with Σ|c_G|² = 1, giving ∫|ψ|² dr = 1.

    AO FT: φ̃(Q) = ∫ e^{-iQ·r} φ(r) dr

    Overlap: ⟨ψ|φ⟩ = (1/√Ω) Σ_G c_G* φ̃(k+G)

    Args:
        psi_list: list of nbnd wavefunctions, each (ngw,) complex
        phi_kg: (ngw, nao) complex - AO in PW basis (FT values)
        cell_volume: cell volume in bohr³

    Returns:
        A: (nbnd, nao) complex
    """
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]

    # Normalization: 1/√Ω
    norm_factor = 1.0 / np.sqrt(cell_volume)

    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        # A[n, μ] = (1/√Ω) Σ_G ψ_n*(G) φ̃_μ(k+G)
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg

    return A


# ============================================================================
# Main comparison
# ============================================================================

print("=" * 70)
print("Comparison: Original diagonalization vs Direct PW overlap")
print("=" * 70)

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'

# Parameters
nbnd = 36
ecut = 30  # Ry, must match QE calculation

# Step 1: Get structure
print("\n[1] Loading structure...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))  # In bohr³
print(f"    Lattice (bohr):\n{rprim}")
print(f"    Cell volume: {cell_volume:.4f} bohr³")

# Load HPRO structure for AO data
structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Step 2: Run original diagonalization method
print("\n[2] Running original diagonalization method...")
kernel = LCAODiagKernel()
kernel.setk([[0.0, 0.0, 0.0]], [1], ['Γ'])
kernel.load_deeph_mats('./')
kernel.diag(nbnd=nbnd, efermi=None)

eigs_diag = kernel.eigs[0, :nbnd]  # Eigenvalues from diagonalization
A_diag = kernel.wfnao[0]  # Eigenvectors (nbnd, nao)
H_loc_diag = compute_local_h_band_ri(eigs_diag, A_diag)

print(f"    Eigenvalues from diag (first 5): {eigs_diag[:5] * hartree2ev} eV")

# Step 3: Direct PW method
print("\n[3] Computing A matrix from direct PW overlap...")

# Find the wfc file for k-point 1 (Γ point)
wfc_path = f'{bands_save_dir}/wfc1.dat'

# Read wavefunction
print(f"    Reading {wfc_path}...")
try:
    psi_list, miller, b_vecs = read_wfc_qe(wfc_path, nbnd)
    ngw = miller.shape[1]
    print(f"    Number of G-vectors: {ngw}")
    print(f"    Number of bands read: {len(psi_list)}")

    # Read eigenvalues from XML
    eigs_xml, kpt = parse_eigenvalues_xml(xml_path, ik=1)
    eigs_pw = eigs_xml[:nbnd]  # Already in Hartree
    print(f"    k-point from XML: {kpt}")
    print(f"    Eigenvalues from XML (first 5): {eigs_pw[:5] * hartree2ev} eV")

    # Compute AO in PW basis
    print("    Computing AO functions in PW basis...")
    phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)
    print(f"    phi_kg shape: {phi_kg.shape}")

    # Compute overlap matrix A = ⟨ψ|φ⟩
    print("    Computing overlap matrix A...")
    A_pw = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)
    print(f"    A_pw shape: {A_pw.shape}")

    # Compute H_loc from PW method
    H_loc_pw = compute_local_h_band_ri(eigs_pw, A_pw)

    pw_success = True

except Exception as e:
    print(f"    ERROR reading wfc file: {e}")
    import traceback
    traceback.print_exc()
    pw_success = False

# Step 4: Comparison
print("\n" + "=" * 70)
print("[4] Comparison")
print("=" * 70)

if pw_success:
    # Compare eigenvalues
    print("\nEigenvalue comparison (eV):")
    print(f"  From diagonalization: {eigs_diag[:5] * hartree2ev}")
    print(f"  From XML file:        {eigs_pw[:5] * hartree2ev}")
    eig_diff = np.abs(eigs_diag - eigs_pw)
    print(f"  Max difference: {np.max(eig_diff) * hartree2ev:.6f} eV")

    # Compare A matrices
    print("\nA matrix comparison:")
    # A matrices may differ by phase, so compare |A|
    A_diag_norm = np.linalg.norm(A_diag, 'fro')
    A_pw_norm = np.linalg.norm(A_pw, 'fro')
    print(f"  ||A_diag||_F = {A_diag_norm:.6f}")
    print(f"  ||A_pw||_F   = {A_pw_norm:.6f}")

    # Compare H_loc
    print("\nH_loc comparison:")
    H_loc_diag_norm = np.linalg.norm(H_loc_diag, 'fro')
    H_loc_pw_norm = np.linalg.norm(H_loc_pw, 'fro')
    diff_norm = np.linalg.norm(H_loc_diag - H_loc_pw, 'fro')

    print(f"  ||H_loc_diag||_F = {H_loc_diag_norm:.6f} Ha")
    print(f"  ||H_loc_pw||_F   = {H_loc_pw_norm:.6f} Ha")
    print(f"  ||H_loc_diag - H_loc_pw||_F = {diff_norm:.6f} Ha")
    print(f"  Relative diff = {diff_norm / H_loc_diag_norm:.6e}")

    # Compare with original H from file
    print("\nComparison with original H(k) from file:")
    matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
    matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)
    Hk_orig = matH.r2k(np.array([0., 0., 0.])).toarray()
    Sk_orig = matS.r2k(np.array([0., 0., 0.])).toarray()

    diff_diag = np.linalg.norm(H_loc_diag - Hk_orig, 'fro') / np.linalg.norm(Hk_orig, 'fro')
    diff_pw = np.linalg.norm(H_loc_pw - Hk_orig, 'fro') / np.linalg.norm(Hk_orig, 'fro')

    print(f"  ||H_loc_diag - H_orig|| / ||H_orig|| = {diff_diag:.4f} ({diff_diag*100:.1f}%)")
    print(f"  ||H_loc_pw - H_orig|| / ||H_orig||   = {diff_pw:.4f} ({diff_pw*100:.1f}%)")

    # Compare overlap matrices: S_pw = A_pw† @ A_pw vs S_orig
    print("\nOverlap matrix comparison:")
    S_diag = A_diag.conj().T @ A_diag  # Should give S for S-orthonormal eigenvectors
    S_pw = A_pw.conj().T @ A_pw

    print(f"  ||S_orig||_F = {np.linalg.norm(Sk_orig, 'fro'):.4f}")
    print(f"  ||S_diag = A_diag† A_diag||_F = {np.linalg.norm(S_diag, 'fro'):.4f}")
    print(f"  ||S_pw = A_pw† A_pw||_F = {np.linalg.norm(S_pw, 'fro'):.4f}")

    # For S-orthonormal eigenvectors: A S A† = I
    # So: A† A should be related to S^{-1}
    ortho_diag = A_diag @ Sk_orig @ A_diag.conj().T
    ortho_pw = A_pw @ Sk_orig @ A_pw.conj().T
    print(f"  ||A_diag S A_diag† - I||_F = {np.linalg.norm(ortho_diag - np.eye(nbnd), 'fro'):.6f}")
    print(f"  ||A_pw S A_pw† - I||_F = {np.linalg.norm(ortho_pw - np.eye(nbnd), 'fro'):.6f}")

    # Check eigenvalue shift
    print("\nEigenvalue shift analysis:")
    shift = np.mean(eigs_pw - eigs_diag)
    print(f"  Mean shift (XML - diag): {shift:.6f} Ha = {shift * hartree2ev:.4f} eV")
    print(f"  This might be Fermi energy difference")

else:
    print("\nPW method failed, skipping comparison.")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
