#!/usr/bin/env python
"""
Plot band structure comparison for MoS2.
Compares: DFT (QE), Original reconstruction (eig.dat),
          Band-RI real-space reconstruction (eig_ri.dat)
"""

import numpy as np
import json
import matplotlib.pyplot as plt

from HPRO.constants import hartree2ev


# =============================================================================
# Parameters
# =============================================================================
min_plot_energy = -8
max_plot_energy = 10
fontsize = 16
plot_dpi = 400

# Fermi energies (eV) for each reconstruction
FERMI_ENERGY_ORIG = 4.785239202  # from diag.py / manual
FERMI_ENERGY_RI = None  # will read from eig_ri.dat header or use same

# Paths
band_json_path = '../../bands/MoS2.save/band.json'


# =============================================================================
# Helper functions
# =============================================================================

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


def load_dft_from_bandjson(path):
    """Load DFT eigenvalues and k-path from band.json."""
    with open(path, 'r') as f:
        data = json.load(f)
    eigs = np.array(data['spin_up_energys']).T  # (nk, nbnd)
    kcoords = np.array(data['kpoints_coords'], dtype=float)
    hsk_coords = data['hsk_coords']
    hsk_symbols = data['plot_hsk_symbols']
    return kcoords, eigs, hsk_coords, hsk_symbols


# =============================================================================
# Load data
# =============================================================================
print("Loading DFT from band.json...")
kcoords_dft, eigs_dft, hsk_coords_dft, _ = load_dft_from_bandjson(band_json_path)
nkpt_dft = len(kcoords_dft)
print(f"DFT: {nkpt_dft} k-points, {eigs_dft.shape[1]} bands")

print("Loading Original reconstruction from eig.dat...")
kcoords_orig, eigs_orig, hsk_coords_orig, hsk_symbols = load_from_eigdat('eig.dat', 'lat.dat')
eigs_orig = eigs_orig - FERMI_ENERGY_ORIG
nkpt_orig = len(kcoords_orig)
print(f"Original: {nkpt_orig} k-points, {eigs_orig.shape[1]} bands")

print("Loading Band-RI reconstruction from eig_ri.dat...")
kcoords_ri, eigs_ri, hsk_coords_ri, _ = load_from_eigdat('eig_ri.dat', 'lat.dat')
# Use same Fermi energy as original (same basis, same overlap)
eigs_ri = eigs_ri - FERMI_ENERGY_ORIG
nkpt_ri = len(kcoords_ri)
print(f"Band-RI: {nkpt_ri} k-points, {eigs_ri.shape[1]} bands")

# =============================================================================
# Scale k-coordinates for consistent plotting
# =============================================================================
x_max = kcoords_orig[-1]
kcoords_dft_scaled = kcoords_dft * (x_max / kcoords_dft[-1])
kcoords_ri_scaled = kcoords_ri * (x_max / kcoords_ri[-1])

nbnd_plot = min(30, eigs_orig.shape[1], eigs_ri.shape[1])

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

for hsk in hsk_coords_orig:
    ax.axvline(hsk, color='black', linewidth=0.7)
ax.axhline(0.0, color='black', linestyle='dashed', linewidth=0.7)

for band_i in range(nbnd_plot):
    label_dft = 'DFT (QE)' if band_i == 0 else None
    ax.plot(kcoords_dft_scaled, eigs_dft[:, band_i], 'r-', linewidth=1.5, label=label_dft, zorder=3)

    label_orig = 'Original' if band_i == 0 else None
    ax.plot(kcoords_orig, eigs_orig[:, band_i], 'b--', linewidth=1.2, label=label_orig, zorder=2)

    label_ri = 'Band-RI (real)' if band_i == 0 else None
    ax.scatter(kcoords_ri_scaled, eigs_ri[:, band_i], c='green', s=3, label=label_ri, zorder=4)

ax.legend(loc='upper right', fontsize=0.75*fontsize)
ax.set_title('MoS2 Band Structure', fontsize=fontsize)

plt.tight_layout()
plt.savefig('band_real.png', dpi=plot_dpi)
plt.savefig('band_real.svg', transparent=True)
print("Saved band_real.png and band_real.svg")

# =============================================================================
# Print MAE summary
# =============================================================================
print("\n" + "="*60)
print("MAE vs DFT (meV) - approximate, different k-grids")
print("="*60)

nk_compare = min(nkpt_orig, nkpt_dft, nkpt_ri)
for n in [4, 8, 16, 20]:
    if n > eigs_orig.shape[1] or n > eigs_ri.shape[1] or n > eigs_dft.shape[1]:
        continue
    mae_orig = np.mean(np.abs(eigs_orig[:nk_compare, :n] - eigs_dft[:nk_compare, :n])) * 1000
    mae_ri = np.mean(np.abs(eigs_ri[:nk_compare, :n] - eigs_dft[:nk_compare, :n])) * 1000
    print(f"Lowest {n:2d} bands: Original = {mae_orig:7.1f} meV, Band-RI = {mae_ri:7.1f} meV")
