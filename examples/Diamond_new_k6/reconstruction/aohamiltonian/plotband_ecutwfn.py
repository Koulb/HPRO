#!/usr/bin/env python
"""
ecutwfn convergence test for Diamond reconstruction.

Runs PW2AOkernel with different ecutwfn values and compares the resulting
band structure against QE reference. Tests how the plane-wave cutoff for
overlap/Vnl integrals affects reconstruction quality.
"""

import os
import sys
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from HPRO import PW2AOkernel
from HPRO.deephio import load_deeph_HS
from HPRO.constants import hartree2ev


# =============================================================================
# Parameters
# =============================================================================
ecutwfn_values = [20, 30, 40, 50, 60, 80]
nbnd_plot = 20
vbm_idx = 3  # Diamond: 4 occupied bands

# Paths (relative to this script's location: reconstruction/aohamiltonian/)
bands_xml = '../../bands/diamond.save/data-file-schema.xml'
recon_dir = '..'           # reconstruction/ directory (where calc.py lives)
test_base = '_ecutwfn_test'  # temp directory for test outputs

# Band path (same as diag.py)
kpath = [[0.000, 0.000, 0.000],
         [0.500, 0.000, 0.500],
         [0.500, 0.250, 0.750],
         [0.500, 0.500, 0.500],
         [0.000, 0.000, 0.000]]
kpath_npts = [20, 15, 15, 20, 1]
kpath_labels = ['Γ', 'X', 'W', 'L', 'Γ']


# =============================================================================
# Helper functions
# =============================================================================

def parse_kpoints_and_eigs_xml(xml_path):
    tree = ET.parse(xml_path)
    kpoints_cart = []
    eigenvalues = []
    for ks_energies in tree.iter('ks_energies'):
        kpt = np.array([float(x) for x in ks_energies.find('k_point').text.split()])
        kpoints_cart.append(kpt)
        eigs = np.array([float(x) for x in ks_energies.find('eigenvalues').text.split()])
        eigenvalues.append(eigs)
    fermi_elem = tree.find('.//fermi_energy')
    fermi_ha = float(fermi_elem.text) if fermi_elem is not None else 0.0
    return kpoints_cart, eigenvalues, fermi_ha


def get_structure_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cell_elem = root.find('.//atomic_structure/cell')
    a1 = np.array([float(x) for x in cell_elem.find('a1').text.split()])
    a2 = np.array([float(x) for x in cell_elem.find('a2').text.split()])
    a3 = np.array([float(x) for x in cell_elem.find('a3').text.split()])
    rprim = np.array([a1, a2, a3])
    return rprim


def generate_kpath(kpath, kpath_npts):
    """Generate k-points along band path (crystal coords, like diag.py)."""
    kpts = []
    hsk_positions = []
    for iseg in range(len(kpath) - 1):
        kstart = np.array(kpath[iseg])
        kend = np.array(kpath[iseg + 1])
        for i in range(kpath_npts[iseg]):
            t = i / kpath_npts[iseg]
            kpts.append(kstart + t * (kend - kstart))
            if i == 0:
                hsk_positions.append(len(kpts) - 1)
    kpts.append(np.array(kpath[-1]))
    hsk_positions.append(len(kpts) - 1)
    return np.array(kpts), hsk_positions


def diagonalize_band_path(matH, matS, kpts, nbnd):
    """Diagonalize H(k) along band path, return eigenvalues in eV."""
    nk = len(kpts)
    eigs = np.zeros((nk, nbnd))
    for ik in range(nk):
        Hk = matH.r2k(kpts[ik]).toarray()
        Sk = matS.r2k(kpts[ik]).toarray()
        Hk = 0.5 * (Hk + Hk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)
        eigs_k, _ = eigh(Hk, Sk)
        eigs[ik] = eigs_k[:nbnd] * hartree2ev
    return eigs


def load_from_eigdat(eig_path, lat_path):
    """Load eigenvalues from eig.dat."""
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


# =============================================================================
# Load QE reference
# =============================================================================
print("=" * 70)
print("ecutwfn Convergence Test for Diamond Reconstruction")
print("=" * 70)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("\n[1] Loading QE reference from bands XML...")
rprim = get_structure_from_xml(bands_xml)
kpoints_cart, qe_eigenvalues, fermi_ha = parse_kpoints_and_eigs_xml(bands_xml)
FERMI_ENERGY_EV = fermi_ha * hartree2ev
nkpt_dft = len(kpoints_cart)

eigs_dft = np.zeros((nkpt_dft, nbnd_plot))
for ik in range(nkpt_dft):
    eigs_dft[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev
eigs_dft -= FERMI_ENERGY_EV

# DFT k-path coordinates
kpoints_cart_arr = np.array(kpoints_cart)
dis_dft = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords_dft = np.concatenate(([0.0], np.cumsum(dis_dft)))
x_max_dft = kcoords_dft[-1]

# HSK points
hsk_idcs_dft = [0]
for i in range(nkpt_dft - 3):
    x1, x2, x3 = kpoints_cart_arr[i], kpoints_cart_arr[i+1], kpoints_cart_arr[i+2]
    if np.sum(np.power(np.cross(x1-x2, x2-x3), 2)) > 1e-15:
        hsk_idcs_dft.append(i+1)
hsk_idcs_dft.append(nkpt_dft - 1)
hsk_coords_dft = [kcoords_dft[i] for i in hsk_idcs_dft]

# VBM shift for DFT
shift_dft = -eigs_dft[:, vbm_idx].max()
eigs_dft_vbm = eigs_dft + shift_dft

print(f"    {nkpt_dft} k-points, Fermi = {FERMI_ENERGY_EV:.4f} eV")

# =============================================================================
# Run reconstruction for each ecutwfn
# =============================================================================
print(f"\n[2] Running reconstructions for ecutwfn = {ecutwfn_values}...")

os.makedirs(test_base, exist_ok=True)

results = {}  # ecutwfn -> {eigs, mae_4, mae_8, mae_16, kcoords}

for ecutwfn in ecutwfn_values:
    outdir = os.path.join(test_base, f'ecut{ecutwfn}')
    print(f"\n--- ecutwfn = {ecutwfn} Ry ---")

    # Run PW2AOkernel from the reconstruction/ directory
    orig_dir = os.getcwd()
    os.chdir(recon_dir)
    try:
        out_full = os.path.join('aohamiltonian', test_base, f'ecut{ecutwfn}')
        kernel = PW2AOkernel(
            lcao_interface='siesta',
            lcaodata_root='../aobasis',
            hrdata_interface='qe-bgw',
            vscdir='../scf/VSC',
            upfdir='../pseudos',
            ecutwfn=ecutwfn
        )
        kernel.run_pw2ao_rs(out_full)
    finally:
        os.chdir(orig_dir)

    # Load the result
    matH = load_deeph_HS(outdir, 'hamiltonians.h5', energy_unit=True)
    matS = load_deeph_HS(outdir, 'overlaps.h5', energy_unit=False)
    matH.hermitianize()
    matS.hermitianize()

    # Diagonalize along band path (same as diag.py)
    kpts, hsk_pos = generate_kpath(kpath, kpath_npts)
    eigs = diagonalize_band_path(matH, matS, kpts, nbnd_plot)

    # Compute k-path coordinates for eig data
    bohr2ang = 0.5291772105638411
    rprim_lat = np.loadtxt(os.path.join(outdir, 'lat.dat')).T / bohr2ang
    gprim_lat = np.linalg.inv(rprim_lat.T)
    kcart = kpts @ gprim_lat
    dis = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
    kcoords = np.concatenate(([0.0], np.cumsum(dis)))

    # VBM align
    shift = -eigs[:, vbm_idx].max()
    eigs_vbm = eigs + shift

    # Scale k-coordinates to match DFT
    kcoords_scaled = kcoords * (x_max_dft / kcoords[-1])

    # Compute MAE vs DFT
    nk_compare = min(nkpt_dft, len(kpts))
    mae = {}
    for n in [4, 8, 16]:
        if n <= nbnd_plot:
            mae[n] = np.mean(np.abs(eigs_vbm[:nk_compare, :n] - eigs_dft_vbm[:nk_compare, :n])) * 1000

    results[ecutwfn] = {
        'eigs_vbm': eigs_vbm,
        'kcoords_scaled': kcoords_scaled,
        'mae': mae,
    }

    print(f"    MAE (4 bands): {mae[4]:.1f} meV")
    print(f"    MAE (8 bands): {mae[8]:.1f} meV")

# =============================================================================
# Also load baseline (current ecutwfn=30 from symlinked hamiltonians.h5)
# =============================================================================
print("\n[3] Loading baseline (current eig.dat, ecutwfn=30)...")
kcoords_base, eigs_base, hsk_coords_base, _ = load_from_eigdat('eig.dat', 'lat.dat')
shift_base = -eigs_base[:, vbm_idx].max()
eigs_base_vbm = eigs_base + shift_base
kcoords_base_scaled = kcoords_base * (x_max_dft / kcoords_base[-1])

nk_base = min(nkpt_dft, len(kcoords_base))
mae_base = {}
for n in [4, 8, 16]:
    mae_base[n] = np.mean(np.abs(eigs_base_vbm[:nk_base, :n] - eigs_dft_vbm[:nk_base, :n])) * 1000
print(f"    Baseline MAE (4 bands): {mae_base[4]:.1f} meV")

# =============================================================================
# Print summary table
# =============================================================================
print("\n" + "=" * 70)
print("ecutwfn Convergence Summary")
print("=" * 70)
print(f"{'ecutwfn (Ry)':>14} {'MAE 4 bands (meV)':>20} {'MAE 8 bands (meV)':>20} {'MAE 16 bands (meV)':>21}")
print("-" * 77)
print(f"{'baseline (30)':>14} {mae_base[4]:20.1f} {mae_base[8]:20.1f} {mae_base[16]:21.1f}")
for ec in ecutwfn_values:
    r = results[ec]
    print(f"{ec:14d} {r['mae'][4]:20.1f} {r['mae'][8]:20.1f} {r['mae'][16]:21.1f}")

# =============================================================================
# Create plots
# =============================================================================
print("\n[4] Creating plots...")
plt.switch_backend('agg')

fontsize = 14
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# --- Plot 1: MAE convergence vs ecutwfn ---
ax1 = axes[0]
for n, color, marker in [(4, 'tab:blue', 'o'), (8, 'tab:orange', 's'), (16, 'tab:green', '^')]:
    mae_vals = [results[ec]['mae'][n] for ec in ecutwfn_values]
    ax1.plot(ecutwfn_values, mae_vals, f'{marker}-', color=color, linewidth=2,
             markersize=8, label=f'{n} bands')
    # Baseline horizontal line
    ax1.axhline(mae_base[n], color=color, linestyle=':', alpha=0.5)

ax1.set_xlabel('ecutwfn (Ry)', fontsize=fontsize)
ax1.set_ylabel('MAE vs QE (meV)', fontsize=fontsize)
ax1.set_title('MAE Convergence', fontsize=fontsize)
ax1.legend(fontsize=0.8*fontsize)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# --- Plot 2: Band structure for lowest and highest ecutwfn ---
ax2 = axes[1]
ec_lo = ecutwfn_values[0]
ec_hi = ecutwfn_values[-1]
ax2.set_xlim(0, x_max_dft)
ax2.set_ylim(-15, 30)
ax2.set_ylabel('Energy (eV)', fontsize=fontsize)
ax2.set_xticks(hsk_coords_dft)
ax2.set_xticklabels(kpath_labels, fontsize=fontsize)
for hsk in hsk_coords_dft:
    ax2.axvline(hsk, color='black', linewidth=0.5)
ax2.axhline(0.0, color='black', linestyle='dashed', linewidth=0.5)

for b in range(min(nbnd_plot, 8)):
    lbl_dft = 'DFT (QE)' if b == 0 else None
    ax2.plot(kcoords_dft, eigs_dft_vbm[:, b], 'r-', linewidth=1.5, label=lbl_dft, zorder=3)
    lbl_lo = f'ecut={ec_lo}' if b == 0 else None
    ax2.plot(results[ec_lo]['kcoords_scaled'], results[ec_lo]['eigs_vbm'][:, b],
             'b--', linewidth=1.0, label=lbl_lo, zorder=2, alpha=0.7)
    lbl_hi = f'ecut={ec_hi}' if b == 0 else None
    ax2.plot(results[ec_hi]['kcoords_scaled'], results[ec_hi]['eigs_vbm'][:, b],
             'g:', linewidth=1.5, label=lbl_hi, zorder=2)

ax2.legend(fontsize=0.7*fontsize, loc='upper right')
ax2.set_title(f'Bands: ecut={ec_lo} vs {ec_hi}', fontsize=fontsize)

# --- Plot 3: Per-band MAE for selected ecutwfn ---
ax3 = axes[2]
n_bands_show = min(8, nbnd_plot)
x_bar = np.arange(n_bands_show)
width = 0.8 / len(ecutwfn_values)

for idx, ec in enumerate(ecutwfn_values):
    r = results[ec]
    mae_per_band = [np.mean(np.abs(r['eigs_vbm'][:nk_compare, b] - eigs_dft_vbm[:nk_compare, b])) * 1000
                    for b in range(n_bands_show)]
    offset = (idx - len(ecutwfn_values)/2 + 0.5) * width
    ax3.bar(x_bar + offset, mae_per_band, width, label=f'ecut={ec}', alpha=0.8)

ax3.set_xlabel('Band index', fontsize=fontsize)
ax3.set_ylabel('MAE (meV)', fontsize=fontsize)
ax3.set_title('Per-band MAE', fontsize=fontsize)
ax3.set_xticks(x_bar)
ax3.legend(fontsize=0.55*fontsize, ncol=2)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ecutwfn_convergence.png', dpi=300)
print("    Saved ecutwfn_convergence.png")

# =============================================================================
# Clean up temp directories
# =============================================================================
print("\n[5] Cleaning up temporary files...")
shutil.rmtree(test_base)
print("    Done.")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

best_ec = min(ecutwfn_values, key=lambda ec: results[ec]['mae'][4])
worst_ec = max(ecutwfn_values, key=lambda ec: results[ec]['mae'][4])
improvement = results[worst_ec]['mae'][4] - results[best_ec]['mae'][4]
print(f"  Best ecutwfn for 4 bands:  {best_ec} Ry (MAE = {results[best_ec]['mae'][4]:.1f} meV)")
print(f"  Worst ecutwfn for 4 bands: {worst_ec} Ry (MAE = {results[worst_ec]['mae'][4]:.1f} meV)")
print(f"  Improvement range:         {improvement:.1f} meV")
print(f"  Baseline (eig.dat):        {mae_base[4]:.1f} meV")
