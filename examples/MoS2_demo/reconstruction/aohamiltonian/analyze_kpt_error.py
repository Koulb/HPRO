"""
Analyze the k-point dependent error in band-RI reconstruction.
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import FortranFile
import matplotlib.pyplot as plt

from HPRO.deephio import load_deeph_HS
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


def parse_all_kpoints_xml(xml_path):
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
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg
    return A


# ============================================================================
# Main analysis
# ============================================================================

print("=" * 70)
print("Analyzing k-point dependent error in band-RI")
print("=" * 70)

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'

# Load data
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)

matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

kpoints, all_eigenvalues = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)

# Compute errors and analyze
print(f"\nAnalyzing {nkpt} k-points...")

k_norms = []
errors_100 = []
A_norms = []
projection_completeness = []

for ik in range(nkpt):
    kpt = kpoints[ik]
    eigs_pw = all_eigenvalues[ik][:100]

    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller = read_wfc_qe(wfc_path, 100)
    except:
        continue

    phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, 30)
    A_pw = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)
    H_loc_pw = compute_local_h_band_ri(eigs_pw, A_pw)

    Hk_orig = matH.r2k(kpt).toarray()
    Sk_orig = matS.r2k(kpt).toarray()

    H_orig_norm = np.linalg.norm(Hk_orig, 'fro')
    error = np.linalg.norm(H_loc_pw - Hk_orig, 'fro') / H_orig_norm

    # Analyze projection quality
    # A A† should be close to I if projection is complete
    proj_matrix = A_pw @ Sk_orig @ A_pw.conj().T
    proj_trace = np.trace(proj_matrix).real
    proj_norm = np.linalg.norm(proj_matrix, 'fro')

    k_cart = gprim.T @ kpt
    k_norm = np.linalg.norm(k_cart)

    k_norms.append(k_norm)
    errors_100.append(error)
    A_norms.append(np.linalg.norm(A_pw, 'fro'))
    projection_completeness.append(proj_trace / 100)  # Should be ~1 if complete

k_norms = np.array(k_norms)
errors_100 = np.array(errors_100)
A_norms = np.array(A_norms)
projection_completeness = np.array(projection_completeness)

print("\nCorrelation analysis:")
print(f"  Correlation(|k|, error): {np.corrcoef(k_norms, errors_100)[0,1]:.4f}")
print(f"  Correlation(||A||, error): {np.corrcoef(A_norms, errors_100)[0,1]:.4f}")
print(f"  Correlation(proj_completeness, error): {np.corrcoef(projection_completeness, errors_100)[0,1]:.4f}")

print("\nK-points with lowest error:")
sorted_idx = np.argsort(errors_100)
for i in sorted_idx[:5]:
    kpt = kpoints[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - error = {errors_100[i]*100:.2f}%, |k| = {k_norms[i]:.4f}, proj = {projection_completeness[i]:.3f}")

print("\nK-points with highest error:")
for i in sorted_idx[-5:]:
    kpt = kpoints[i]
    print(f"  k{i+1}: ({kpt[0]:.3f}, {kpt[1]:.3f}, {kpt[2]:.3f}) - error = {errors_100[i]*100:.2f}%, |k| = {k_norms[i]:.4f}, proj = {projection_completeness[i]:.3f}")

# Check if projection completeness correlates with error
print("\n" + "=" * 70)
print("Projection completeness analysis")
print("=" * 70)
print("Tr(A S A†) / nbnd should be ~1 if bands span AO space well")
print(f"  Min: {projection_completeness.min():.4f}")
print(f"  Max: {projection_completeness.max():.4f}")
print(f"  Mean: {projection_completeness.mean():.4f}")

# Save plot
try:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(k_norms, errors_100*100)
    axes[0].set_xlabel('|k| (bohr⁻¹)')
    axes[0].set_ylabel('Error (%)')
    axes[0].set_title('Error vs |k|')

    axes[1].scatter(projection_completeness, errors_100*100)
    axes[1].set_xlabel('Projection completeness')
    axes[1].set_ylabel('Error (%)')
    axes[1].set_title('Error vs Projection')

    axes[2].scatter(A_norms, errors_100*100)
    axes[2].set_xlabel('||A||_F')
    axes[2].set_ylabel('Error (%)')
    axes[2].set_title('Error vs ||A||')

    plt.tight_layout()
    plt.savefig('kpt_error_analysis.png', dpi=150)
    print("\nPlot saved to kpt_error_analysis.png")
except Exception as e:
    print(f"\nCould not save plot: {e}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
