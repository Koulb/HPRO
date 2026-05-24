#!/usr/bin/env python
"""
Plot band structure comparison for Diamond 2x2x2 supercell.
Compares: DFT (QE bands), Original reconstruction (eig.dat),
          Band-RI real-space reconstruction (eig_ri.dat)
"""

import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from HPRO.constants import hartree2ev


def load_from_eigdat(eig_path, lat_path):
    """Load eigenvalues and k-path from eig.dat format."""
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


# =============================================================================
# Parameters
# =============================================================================
min_plot_energy = -15
max_plot_energy = 30
fontsize = 16
plot_dpi = 400
# VBM at band 31 (index 31) for 16-atom supercell (64 electrons, 32 occupied bands)
vbm_idx = 31
nbnd_plot = 40

# Paths
bands_save_dir = '../../bands/diamond.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'

# =============================================================================
# Load DFT data from QE bands XML
# =============================================================================
print("Loading DFT data from QE bands XML...")

kpoints_cart, qe_eigenvalues, fermi_ha = parse_kpoints_and_eigs_xml(xml_path)
FERMI_ENERGY_EV = fermi_ha * hartree2ev
nkpt_dft = len(kpoints_cart)

eigs_dft = np.zeros((nkpt_dft, nbnd_plot))
for ik in range(nkpt_dft):
    eigs_dft[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev
eigs_dft -= FERMI_ENERGY_EV

# DFT k-coordinates
kpoints_cart_arr = np.array(kpoints_cart)
dis = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords_dft = np.concatenate(([0.0], np.cumsum(dis)))

# Find HSK points
hsk_idcs_dft = [0]
for i in range(nkpt_dft - 3):
    x1, x2, x3 = kpoints_cart_arr[i], kpoints_cart_arr[i+1], kpoints_cart_arr[i+2]
    is_corner = np.sum(np.power(np.cross(x1-x2, x2-x3), 2)) > 1e-15
    if is_corner:
        hsk_idcs_dft.append(i+1)
hsk_idcs_dft.append(nkpt_dft - 1)
hsk_coords_dft = [kcoords_dft[i] for i in hsk_idcs_dft]
hsk_symbols = ['Γ', 'X', 'W', 'L', 'Γ']

print(f"DFT: {nkpt_dft} k-points, Fermi = {FERMI_ENERGY_EV:.4f} eV")

# =============================================================================
# Load Original reconstruction from eig.dat
# =============================================================================
print("Loading Original reconstruction from eig.dat...")
kcoords_orig, eigs_orig, hsk_coords_orig, _ = load_from_eigdat('eig.dat', 'lat.dat')
nkpt_orig = len(kcoords_orig)
print(f"Original: {nkpt_orig} k-points, {eigs_orig.shape[1]} bands")

# =============================================================================
# Load Band-RI reconstruction from eig_ri.dat
# =============================================================================
print("Loading Band-RI reconstruction from eig_ri.dat...")
kcoords_ri, eigs_ri, _, _ = load_from_eigdat('eig_ri.dat', 'lat.dat')
nkpt_ri = len(kcoords_ri)
print(f"Band-RI: {nkpt_ri} k-points, {eigs_ri.shape[1]} bands")

# =============================================================================
# Align energies using VBM
# =============================================================================
shift_dft = -eigs_dft[:, vbm_idx].max()
shift_orig = -eigs_orig[:, vbm_idx].max()
shift_ri = -eigs_ri[:, vbm_idx].max()

eigs_dft_shifted = eigs_dft + shift_dft
eigs_orig_shifted = eigs_orig + shift_orig
eigs_ri_shifted = eigs_ri + shift_ri

# Scale k-coordinates for consistent plotting
x_max = kcoords_dft[-1]
kcoords_orig_scaled = kcoords_orig * (x_max / kcoords_orig[-1])
kcoords_ri_scaled = kcoords_ri * (x_max / kcoords_ri[-1])

nbnd_plot = min(nbnd_plot, eigs_orig.shape[1], eigs_ri.shape[1])

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
ax.set_xticks(hsk_coords_dft)
ax.set_xticklabels(hsk_symbols, fontsize=fontsize)
ax.tick_params('y', labelsize=0.85*fontsize)

for hsk in hsk_coords_dft:
    ax.axvline(hsk, color='black', linewidth=0.7)
ax.axhline(0.0, color='black', linestyle='dashed', linewidth=0.7)

for band_i in range(nbnd_plot):
    label_dft = 'DFT (QE)' if band_i == 0 else None
    ax.plot(kcoords_dft, eigs_dft_shifted[:, band_i], 'r-', linewidth=1.5, label=label_dft, zorder=3)

    label_orig = 'Original' if band_i == 0 else None
    ax.plot(kcoords_orig_scaled, eigs_orig_shifted[:, band_i], 'b--', linewidth=1.2, label=label_orig, zorder=2)

    label_ri = 'Band-RI' if band_i == 0 else None
    ax.scatter(kcoords_ri_scaled, eigs_ri_shifted[:, band_i], c='green', s=3, label=label_ri, zorder=4)

ax.legend(loc='upper right', fontsize=0.75*fontsize)
ax.set_title('Diamond 2x2x2 Supercell Band Structure', fontsize=fontsize)

plt.tight_layout()
plt.savefig('band_real.png', dpi=plot_dpi)
plt.savefig('band_real.svg', transparent=True)
print("Saved band_real.png and band_real.svg")

# =============================================================================
# Print MAE summary (VBM-aligned)
# =============================================================================
print("\n" + "="*60)
print("MAE vs DFT (meV) - VBM-aligned")
print("="*60)

nk_compare = min(nkpt_dft, nkpt_orig, nkpt_ri)
for n in [32, 40]:
    if n > nbnd_plot:
        continue
    mae_orig = np.mean(np.abs(eigs_orig_shifted[:nk_compare, :n] - eigs_dft_shifted[:nk_compare, :n])) * 1000
    mae_ri = np.mean(np.abs(eigs_ri_shifted[:nk_compare, :n] - eigs_dft_shifted[:nk_compare, :n])) * 1000
    print(f"Lowest {n:2d} bands: Original = {mae_orig:7.1f} meV, Band-RI = {mae_ri:7.1f} meV")
