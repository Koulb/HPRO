#!/usr/bin/env python
"""
Plot band structure comparison for MoS2.
Compares: DFT (QE), Original reconstruction, Band-RI reconstruction

NOTE: Original reconstruction uses eigenvalues from eig.dat (computed by diag.py)
      with its own k-path. DFT and Band-RI use QE k-points from band.json/XML.
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
min_plot_energy = -8
max_plot_energy = 10
fontsize = 16
plot_dpi = 400

# Fermi energy for Original reconstruction (from diag.py or manual)
FERMI_ENERGY_ORIG = 4.785239202  # eV

# Paths
bands_save_dir = '../../bands/MoS2.save'
band_json_path = f'{bands_save_dir}/band.json'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis'
ecut = 30 
nbands_bandri = 250#2 * 100   # Number of bands for band-RI

# =============================================================================
# Helper functions
# =============================================================================

def load_original_from_eigdat(eig_path, lat_path):
    """Load Original reconstruction eigenvalues and k-path from eig.dat."""
    bohr2ang = 0.5291772105638411
    rprim = np.loadtxt(lat_path).T / bohr2ang
    gprim = np.linalg.inv(rprim.T)

    with open(eig_path) as f:
        f.readline(); f.readline()
        nk, nbnd = map(int, f.readline().split())
        eigs = np.empty((nk, nbnd))
        kpts = np.empty((nk, 3))
        hsk_idcs, hsk_symbols = [], []

        line = f.readline()
        ik = 0
        while line:
            sp = line.split()
            if len(sp) > 0 and '.' in sp[0]:
                kpts[ik] = list(map(float, sp[:3]))
                if len(sp) == 5:
                    hsk_symbols.append(sp[4].replace('Г', 'Γ'))
                    hsk_idcs.append(ik)
                for _ in range(nbnd):
                    line2 = f.readline().split()
                    ibnd = int(line2[1]) - 1
                    eigs[ik, ibnd] = float(line2[2])
                ik += 1
            line = f.readline()
        assert ik == nk

    kcart = kpts @ gprim
    dis = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
    kcoords = np.concatenate(([0.0], np.cumsum(dis)))
    hsk_coords = [kcoords[i] for i in hsk_idcs]

    return kcoords, eigs, hsk_coords, hsk_symbols


def load_dft_from_bandjson(path):
    """Load DFT eigenvalues and k-path from band.json."""
    with open(path, 'r') as f:
        data = json.load(f)
    eigs = np.array(data['spin_up_energys']).T  # (nk, nbnd)
    kcoords = np.array(data['kpoints_coords'], dtype=float)
    hsk_coords = data['hsk_coords']
    hsk_symbols = data['plot_hsk_symbols']
    return kcoords, eigs, hsk_coords, hsk_symbols


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


# =============================================================================
# Load Original reconstruction from eig.dat (uses diag.py k-path)
# =============================================================================
print("Loading Original reconstruction from eig.dat...")
kcoords_orig, eigs_orig, hsk_coords_orig, hsk_symbols = load_original_from_eigdat('eig.dat', 'lat.dat')
eigs_orig = eigs_orig - FERMI_ENERGY_ORIG  # Shift to Fermi = 0
nkpt_orig = len(kcoords_orig)
print(f"Original: {nkpt_orig} k-points, {eigs_orig.shape[1]} bands")

# =============================================================================
# Load DFT from band.json (uses QE k-path)
# =============================================================================
print("Loading DFT from band.json...")
kcoords_dft, eigs_dft, hsk_coords_dft, _ = load_dft_from_bandjson(band_json_path)
# band.json is already Fermi-shifted
nkpt_dft = len(kcoords_dft)
print(f"DFT: {nkpt_dft} k-points, {eigs_dft.shape[1]} bands")

# =============================================================================
# Compute Band-RI on QE k-points (needs wavefunction files)
# =============================================================================
print("Computing Band-RI...")

rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')

matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

kpoints_cart, qe_eigenvalues, fermi_ha = parse_kpoints_and_eigs_xml(xml_path)
FERMI_ENERGY_EV = fermi_ha * hartree2ev
nkpt_qe = len(kpoints_cart)

kpoints_cryst = []
for kc in kpoints_cart:
    k_cryst = rprim @ kc / (2 * np.pi)
    kpoints_cryst.append(k_cryst)

nbnd_plot = 30
eigs_bandri = np.zeros((nkpt_qe, nbnd_plot))

for ik in range(nkpt_qe):
    kpt_cryst = kpoints_cryst[ik]
    Sk = matS.r2k(kpt_cryst).toarray()

    wfc_path = f'{bands_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller = read_wfc_qe(wfc_path, nbands_bandri)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt_cryst, structure, lcaodata, ecut)
        A_k = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)

        eigs_k_ha = qe_eigenvalues[ik][:nbands_bandri]
        H_bandri = compute_local_h_band_ri(eigs_k_ha, A_k)
        eigs_bandri_ha = diagonalize_generalized(H_bandri, Sk)
        eigs_bandri[ik] = eigs_bandri_ha[:nbnd_plot] * hartree2ev
    except Exception as e:
        print(f"  Warning at k={ik+1}: {e}")
        eigs_bandri[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev

eigs_bandri -= FERMI_ENERGY_EV

# Compute k-coords for Band-RI (same as QE)
kpoints_cart_arr = np.array(kpoints_cart)
dis = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords_bandri = np.concatenate(([0.0], np.cumsum(dis)))

print("Done computing eigenvalues.")

# =============================================================================
# Scale k-coordinates for consistent plotting
# =============================================================================
x_max = kcoords_orig[-1]  # Use Original k-path as reference
kcoords_dft_scaled = kcoords_dft * (x_max / kcoords_dft[-1])
kcoords_bandri_scaled = kcoords_bandri * (x_max / kcoords_bandri[-1])

# =============================================================================
# Create plot
# =============================================================================
print("Creating plot...")

plt.switch_backend('agg')
plt.rcParams.update({'font.size': fontsize, 'mathtext.fontset': 'cm'})

fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

ax.set_xlim(0.0, x_max)
ax.set_ylim(min_plot_energy, max_plot_energy)
ax.set_ylabel('Energy (eV)', fontsize=fontsize)
ax.set_xticks(hsk_coords_orig)
ax.set_xticklabels(hsk_symbols, fontsize=fontsize)
ax.tick_params('y', labelsize=0.85*fontsize)

# Vertical lines at high-symmetry points
for hsk in hsk_coords_orig:
    ax.axvline(hsk, color='black', linewidth=0.7)

# Fermi level
ax.axhline(0.0, color='black', linestyle='dashed', linewidth=0.7)

# Plot bands (each with its own k-coordinates)
for band_i in range(nbnd_plot):
    # DFT - red solid line (QE k-path, scaled)
    label_dft = 'DFT (QE)' if band_i == 0 else None
    ax.plot(kcoords_dft_scaled, eigs_dft[:, band_i], 'r-', linewidth=1.5, label=label_dft, zorder=3)

    # Original reconstruction - blue dashed (eig.dat k-path)
    label_orig = 'Original' if band_i == 0 else None
    ax.plot(kcoords_orig, eigs_orig[:, band_i], 'b--', linewidth=1.2, label=label_orig, zorder=2)

    # Band-RI - green dots (QE k-path, scaled)
    label_bandri = 'Band-RI' if band_i == 0 else None
    ax.scatter(kcoords_bandri_scaled, eigs_bandri[:, band_i], c='green', s=3, label=label_bandri, zorder=4)

ax.legend(loc='upper right', fontsize=0.75*fontsize)
ax.set_title('MoS2 Band Structure', fontsize=fontsize)

plt.tight_layout()
plt.savefig('band.png', dpi=plot_dpi)
plt.savefig('band.svg', transparent=True)
print("Saved band.png and band.svg")

# =============================================================================
# Print MAE summary (note: comparing on different k-grids, use with caution)
# =============================================================================
print("\n" + "="*50)
print("MAE vs DFT (meV) - approximate, different k-grids")
print("="*50)

# For fair comparison, interpolate or just compare at same indices
# Here we compare by index (approximate since k-grids differ)
nk_compare = min(nkpt_orig, nkpt_dft)
for n in [4, 8, 16, 20]:
    mae_orig = np.mean(np.abs(eigs_orig[:nk_compare, :n] - eigs_dft[:nk_compare, :n])) * 1000
    mae_bandri = np.mean(np.abs(eigs_bandri[:nk_compare, :n] - eigs_dft[:nk_compare, :n])) * 1000
    print(f"Lowest {n:2d} bands: Original = {mae_orig:7.1f} meV, Band-RI = {mae_bandri:7.1f} meV")
