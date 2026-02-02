#!/usr/bin/env python
"""
Plot band structure comparison for Diamond.
Compares: DFT (QE), Original reconstruction, Band-RI reconstruction
"""

import numpy as np
import json
import xml.etree.ElementTree as ET
from scipy.io import FortranFile
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from HPRO.deephio import load_deeph_HS
from HPRO.mathutils import compute_local_h_band_ri
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs
from HPRO.constants import hartree2ev

# =============================================================================
# Parameters
# =============================================================================
min_plot_energy = -15
max_plot_energy = 30
fontsize = 16
plot_dpi = 400

# Fermi energy will be read from QE XML file

# Paths
bands_save_dir = '../../bands/diamond.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis'
ecut = 30
nbands_bandri = 100  # Number of bands for band-RI

# =============================================================================
# Helper functions
# =============================================================================

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


def parse_kpoints_and_eigs_xml(xml_path):
    """Parse k-points, eigenvalues, and Fermi energy from QE XML."""
    tree = ET.parse(xml_path)
    kpoints_cart = []
    eigenvalues = []
    for ks_energies in tree.iter('ks_energies'):
        kpt = np.array([float(x) for x in ks_energies.find('k_point').text.split()])
        kpoints_cart.append(kpt)
        eigs = np.array([float(x) for x in ks_energies.find('eigenvalues').text.split()])
        eigenvalues.append(eigs)
    # Get Fermi energy from XML (in Hartree)
    fermi_elem = tree.find('.//fermi_energy')
    fermi_energy_ha = float(fermi_elem.text) if fermi_elem is not None else 0.0
    return kpoints_cart, eigenvalues, fermi_energy_ha


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


def compute_ao_in_pw_basis(miller, gprim, kpt_cryst, structure, lcaodata, ecut):
    """Compute AO functions in PW basis."""
    ngw = miller.shape[1]
    kgcart = np.zeros((ngw, 3))
    for ig in range(ngw):
        g_cryst = miller[:, ig]
        g_cart = gprim.T @ g_cryst
        k_cart = gprim.T @ kpt_cryst
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
    """Compute A[n,μ] = ⟨ψ_n|φ_μ⟩."""
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg
    return A


def diagonalize_generalized(H, S):
    """Solve H c = ε S c."""
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


def loaddata_json(filepath):
    """Load band.json data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    for key, val in data.items():
        if type(val) is list:
            data[key] = np.array(val)
    return data


# =============================================================================
# Load structure and matrices
# =============================================================================
print("Loading structure and data...")

rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)

# Load H and S from original reconstruction
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get k-points, QE eigenvalues, and Fermi energy from bands calculation
kpoints_cart, qe_eigenvalues, fermi_ha = parse_kpoints_and_eigs_xml(xml_path)
FERMI_ENERGY_EV = fermi_ha * hartree2ev
nkpt = len(kpoints_cart)
print(f"Fermi energy from XML: {FERMI_ENERGY_EV:.6f} eV")

# Convert k-points to crystal coordinates
kpoints_cryst = []
for kc in kpoints_cart:
    k_cryst = rprim @ kc / (2 * np.pi)
    kpoints_cryst.append(k_cryst)

print(f"Number of k-points on band path: {nkpt}")

# =============================================================================
# Compute eigenvalues for all three methods
# =============================================================================
print("Computing eigenvalues...")

nbnd_plot = 20  # Number of bands to plot

# Arrays to store eigenvalues
eigs_dft = np.zeros((nkpt, nbnd_plot))      # QE reference
eigs_original = np.zeros((nkpt, nbnd_plot))  # Original reconstruction
eigs_bandri = np.zeros((nkpt, nbnd_plot))    # Band-RI

for ik in range(nkpt):
    kpt_cryst = kpoints_cryst[ik]

    # DFT eigenvalues (from QE)
    eigs_dft[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev

    # Original reconstruction
    Hk = matH.r2k(kpt_cryst).toarray()
    Sk = matS.r2k(kpt_cryst).toarray()
    eigs_orig_ha = diagonalize_generalized(Hk, Sk)
    eigs_original[ik] = eigs_orig_ha[:nbnd_plot] * hartree2ev

    # Band-RI reconstruction
    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller = read_wfc_qe(wfc_path, nbands_bandri)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt_cryst, structure, lcaodata, ecut)
        A_k = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)

        # Use QE eigenvalues for band-RI
        eigs_k_ha = qe_eigenvalues[ik][:nbands_bandri]
        H_bandri = compute_local_h_band_ri(eigs_k_ha, A_k)
        eigs_bandri_ha = diagonalize_generalized(H_bandri, Sk)
        eigs_bandri[ik] = eigs_bandri_ha[:nbnd_plot] * hartree2ev
    except Exception as e:
        print(f"  Warning at k={ik+1}: {e}")
        eigs_bandri[ik] = eigs_dft[ik]

# Shift all to Fermi energy = 0
eigs_dft -= FERMI_ENERGY_EV
eigs_original -= FERMI_ENERGY_EV
eigs_bandri -= FERMI_ENERGY_EV

print("Done computing eigenvalues.")

# =============================================================================
# Compute k-path coordinates
# =============================================================================
kpoints_cart_arr = np.array(kpoints_cart)
dis = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords = np.concatenate(([0.0], np.cumsum(dis)))

# Find high-symmetry points
hsk_idcs = [0]
for i in range(nkpt - 3):
    x1, x2, x3 = kpoints_cart_arr[i], kpoints_cart_arr[i+1], kpoints_cart_arr[i+2]
    is_corner = np.sum(np.power(np.cross(x1-x2, x2-x3), 2)) > 1e-15
    if is_corner:
        hsk_idcs.append(i+1)
hsk_idcs.append(nkpt - 1)

hsk_coords = [kcoords[i] for i in hsk_idcs]
hsk_symbols = ['Γ', 'X', 'W', 'L', 'Γ']

# =============================================================================
# Create plot
# =============================================================================
print("Creating plot...")

plt.switch_backend('agg')
plt.rcParams.update({'font.size': fontsize, 'mathtext.fontset': 'cm'})

fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

x_min, x_max = 0.0, kcoords[-1]
ax.set_xlim(x_min, x_max)
ax.set_ylim(min_plot_energy, max_plot_energy)
ax.set_ylabel('Energy (eV)', fontsize=fontsize)
ax.set_xticks(hsk_coords)
ax.set_xticklabels(hsk_symbols, fontsize=fontsize)
ax.tick_params('y', labelsize=0.85*fontsize)

# Vertical lines at high-symmetry points
for hsk in hsk_coords:
    ax.axvline(hsk, color='black', linewidth=0.7)

# Fermi level
ax.axhline(0.0, color='black', linestyle='dashed', linewidth=0.7)

shift_ri = -eigs_bandri[:,3].max() 
shift_dft = -eigs_dft[:,3].max()

# Plot bands
for band_i in range(nbnd_plot):
    # DFT - red solid line
    label_dft = 'DFT (QE)' if band_i == 0 else None
    ax.plot(kcoords, eigs_dft[:, band_i] + shift_dft, 'r-', linewidth=1.5, label=label_dft, zorder=3)

    # Original reconstruction - blue dashed
    label_orig = 'Original' if band_i == 0 else None
    ax.plot(kcoords, eigs_original[:, band_i], 'b--', linewidth=1.2, label=label_orig, zorder=2)

    # Band-RI - green dots
    label_bandri = 'Band-RI' if band_i == 0 else None
    ax.scatter(kcoords, eigs_bandri[:, band_i] + shift_ri, c='green', s=3, label=label_bandri, zorder=4)

ax.legend(loc='upper right', fontsize=0.75*fontsize)
ax.set_title('Diamond Band Structure', fontsize=fontsize)

plt.tight_layout()
plt.savefig('band.png', dpi=plot_dpi)
plt.savefig('band.svg', transparent=True)
print("Saved band.png and band.svg")

# =============================================================================
# Print MAE summary
# =============================================================================
print("\n" + "="*50)
print("MAE vs DFT (meV)")
print("="*50)

for n in [4, 8, 16]:
    mae_orig = np.mean(np.abs(eigs_original[:, :n] - eigs_dft[:, :n])) * 1000
    mae_bandri = np.mean(np.abs(eigs_bandri[:, :n] - eigs_dft[:, :n])) * 1000
    print(f"Lowest {n:2d} bands: Original = {mae_orig:7.1f} meV, Band-RI = {mae_bandri:7.1f} meV")
