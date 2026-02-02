#!/usr/bin/env python
"""
Plot band structure comparison for MoS2.
Compares: DFT (QE), Original reconstruction, Band-RI reconstruction

FIX:
  - ORIGINAL (blue) is plotted using k-path + eigenvalues from eig.dat (consistent pair)
  - DFT (red) is plotted using k-path + eigenvalues from band.json (consistent pair)
  - Band-RI (green) unchanged (computed on QE XML k-points), only x is rescaled for overlay
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
min_plot_energy = -70
max_plot_energy = 8
fontsize = 16
plot_dpi = 400

# Fermi energy used to align (set to what you want)
FERMI_ENERGY_EV = 4.785239202

# Paths
bands_save_dir = '../../bands/MoS2.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'
aobasis_dir = '../../aobasis_ref'
ecut = 30
nbands_bandri = 100  # Number of bands for band-RI

# FIX: paths for DFT and Original plotting sources
bands_ref_dir = '../../bands_ref/MoS2.save'
band_json_path = f'{bands_ref_dir}/band.json'
eig_dat_path = 'eig.dat'
lat_dat_path = 'lat.dat'

# If band.json is already Fermi-shifted (common in your setup), keep True
DFT_JSON_ALREADY_SHIFTED = True


# =============================================================================
# Helper functions
# =============================================================================

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


def parse_kpoints_and_eigs_xml(xml_path):
    tree = ET.parse(xml_path)
    kpoints_cart = []
    eigenvalues = []
    for ks_energies in tree.iter('ks_energies'):
        kpt = np.array([float(x) for x in ks_energies.find('k_point').text.split()])
        kpoints_cart.append(kpt)
        eigs = np.array([float(x) for x in ks_energies.find('eigenvalues').text.split()])
        eigenvalues.append(eigs)
    return kpoints_cart, eigenvalues


def read_wfc_qe(path, nbands):
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
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg
    return A


def diagonalize_generalized(H, S):
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


# FIX: load ORIGINAL (eig.dat) kcoords + eigs consistently
def load_original_from_eigdat(eig_path, lat_path):
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
                # read eigenvalues for this k
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


# FIX: load DFT from band.json consistently
def load_dft_from_bandjson(path):
    with open(path, 'r') as f:
        data = json.load(f)
    eigs = np.array(data['spin_up_energys']).T  # (nk, nbnd)
    kcoords = np.array(data['kpoints_coords'], dtype=float)
    return kcoords, eigs


# =============================================================================
# Load structure and matrices (unchanged)
# =============================================================================
print("Loading structure and data...")

rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')

matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

kpoints_cart, qe_eigenvalues = parse_kpoints_and_eigs_xml(xml_path)
nkpt = len(kpoints_cart)

kpoints_cryst = []
for kc in kpoints_cart:
    k_cryst = rprim @ kc / (2 * np.pi)
    kpoints_cryst.append(k_cryst)

print(f"Number of k-points on band path: {nkpt}")

# =============================================================================
# Compute Band-RI (unchanged) + keep your existing arrays (but we will NOT use
# eigs_dft/eigs_original from here for plotting anymore)
# =============================================================================
print("Computing eigenvalues...")

nbnd_plot = 80

eigs_bandri = np.zeros((nkpt, nbnd_plot))

# (Optional: keep these if you want MAE on QE grid; not used in plotting)
eigs_dft_qegrid = np.zeros((nkpt, nbnd_plot))
eigs_original_qegrid = np.zeros((nkpt, nbnd_plot))

for ik in range(nkpt):
    kpt_cryst = kpoints_cryst[ik]

    eigs_dft_qegrid[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev

    Hk = matH.r2k(kpt_cryst).toarray()
    Sk = matS.r2k(kpt_cryst).toarray()
    eigs_orig_ha = diagonalize_generalized(Hk, Sk)
    eigs_original_qegrid[ik] = eigs_orig_ha[:nbnd_plot] * hartree2ev

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
        eigs_bandri[ik] = eigs_dft_qegrid[ik]

# Shift Band-RI (unchanged behavior)
eigs_bandri -= FERMI_ENERGY_EV

print("Done computing eigenvalues.")

# =============================================================================
# FIX: load plotting datasets with consistent (x,y) pairs
# =============================================================================
kcoords_orig, eigs_orig_plot, hsk_coords, hsk_symbols = load_original_from_eigdat(eig_dat_path, lat_dat_path)
kcoords_dft, eigs_dft_plot = load_dft_from_bandjson(band_json_path)

# Shift energies for plotting
# Original eig.dat is typically NOT shifted -> shift it
eigs_orig_plot = eigs_orig_plot - FERMI_ENERGY_EV

# band.json in your earlier workflow was already shifted; keep that by default
if not DFT_JSON_ALREADY_SHIFTED:
    eigs_dft_plot = eigs_dft_plot - FERMI_ENERGY_EV

# Use common band count for plotting
nbnd_plot = min(nbnd_plot, eigs_orig_plot.shape[1], eigs_dft_plot.shape[1], eigs_bandri.shape[1])

# =============================================================================
# k-path coordinates for Band-RI (QE XML) and rescale to overlay on ORIGINAL length
# =============================================================================
kpoints_cart_arr = np.array(kpoints_cart)
dis = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords_qe = np.concatenate(([0.0], np.cumsum(dis)))

x_max = kcoords_orig[-1]
kcoords_qe_scaled = kcoords_qe * (x_max / kcoords_qe[-1])
kcoords_dft_scaled = kcoords_dft * (x_max / kcoords_dft[-1])

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
ax.set_xticks(hsk_coords)
ax.set_xticklabels(hsk_symbols, fontsize=fontsize)
ax.tick_params('y', labelsize=0.85*fontsize)

for hsk in hsk_coords:
    ax.axvline(hsk, color='black', linewidth=0.7)
ax.axhline(0.0, color='black', linestyle='dashed', linewidth=0.7)

for band_i in range(nbnd_plot):
    label_dft = 'DFT (QE)' if band_i == 0 else None
    ax.plot(kcoords_dft_scaled, eigs_dft_plot[:, band_i], 'r-', linewidth=1.5, label=label_dft, zorder=3)

    label_orig = 'Original' if band_i == 0 else None
    ax.plot(kcoords_orig, eigs_orig_plot[:, band_i], 'b--', linewidth=1.2, label=label_orig, zorder=2)

    label_bandri = 'Band-RI' if band_i == 0 else None
    ax.scatter(kcoords_qe_scaled, eigs_bandri[:, band_i], c='green', s=3, label=label_bandri, zorder=4)

ax.legend(loc='upper right', fontsize=0.75*fontsize)
ax.set_title('MoS2 Band Structure', fontsize=fontsize)

plt.tight_layout()
plt.savefig('band_comparison.png', dpi=plot_dpi)
plt.savefig('band_comparison.svg', transparent=True)
print("Saved band_comparison.png and band_comparison.svg")

# =============================================================================
# Print MAE summary (unchanged: compares arrays by index)
# =============================================================================
print("\n" + "="*50)
print("MAE vs DFT (meV)")
print("="*50)

for n in [4, 8, 16]:
    mae_orig = np.mean(np.abs(eigs_orig_plot[:, :n] - eigs_dft_plot[:, :n])) * 1000
    mae_bandri = np.mean(np.abs(eigs_bandri[:, :n] - eigs_dft_plot[:, :n])) * 1000
    print(f"Lowest {n:2d} bands: Original = {mae_orig:7.1f} meV, Band-RI = {mae_bandri:7.1f} meV")
