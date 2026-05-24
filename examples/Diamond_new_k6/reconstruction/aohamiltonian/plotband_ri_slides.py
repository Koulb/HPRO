#!/usr/bin/env python
"""
Plot band structure comparison for Diamond - APS slides style.
Compares: DFT (QE), Original reconstruction (eig.dat),
          Band-RI real-space reconstruction (eig_ri.dat)
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.linalg import eigh
import matplotlib.pyplot as plt

hartree2ev = 27.211386024367243


# =============================================================================
# Style (from plots_aps.ipynb)
# =============================================================================
marp_text_color = "#575279"
color_pbe    = "mediumseagreen"
color_orig   = "#b4637a"
color_ri     = "#286983"
alpha        = 0.8
legend_alpha = 0.5
line_width   = 3
fontsize     = 22

plt.rcParams.update({
    'font.size':        fontsize,
    'mathtext.fontset': 'cm',
    'text.color':       marp_text_color,
    'axes.labelcolor':  marp_text_color,
    'xtick.color':      marp_text_color,
    'ytick.color':      marp_text_color,
    'axes.edgecolor':   marp_text_color,
    'axes.labelpad':    10,
})

# =============================================================================
# Parameters
# =============================================================================
min_plot_energy = -30
max_plot_energy = 30
nbnd_plot = 20
plot_dpi = 300

bands_save_dir = '../../bands/diamond.save'
xml_path = f'{bands_save_dir}/data-file-schema.xml'

output_path = '/home/apolyukhin/git/aps_slides/random_slides/pictures_ml/band-ri.png'


# =============================================================================
# Helper functions
# =============================================================================

def load_from_eigdat(eig_path, lat_path):
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
    tree = ET.parse(xml_path)
    kpoints_cart, eigenvalues = [], []
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
    return rprim, 2 * np.pi * np.linalg.inv(rprim.T)


# =============================================================================
# Load data
# =============================================================================
print("Loading DFT data...")
rprim, gprim = get_structure_from_xml(xml_path)
kpoints_cart, qe_eigenvalues, fermi_ha = parse_kpoints_and_eigs_xml(xml_path)
FERMI_ENERGY_EV = fermi_ha * hartree2ev
nkpt_dft = len(kpoints_cart)

eigs_dft = np.zeros((nkpt_dft, nbnd_plot))
for ik in range(nkpt_dft):
    eigs_dft[ik] = qe_eigenvalues[ik][:nbnd_plot] * hartree2ev
eigs_dft -= FERMI_ENERGY_EV

kpoints_cart_arr = np.array(kpoints_cart)
dis = np.linalg.norm(np.diff(kpoints_cart_arr, axis=0), axis=1)
kcoords_dft = np.concatenate(([0.0], np.cumsum(dis)))

hsk_idcs_dft = [0]
for i in range(nkpt_dft - 3):
    x1, x2, x3 = kpoints_cart_arr[i], kpoints_cart_arr[i+1], kpoints_cart_arr[i+2]
    if np.sum(np.power(np.cross(x1-x2, x2-x3), 2)) > 1e-15:
        hsk_idcs_dft.append(i+1)
hsk_idcs_dft.append(nkpt_dft - 1)
hsk_coords_dft = [kcoords_dft[i] for i in hsk_idcs_dft]
hsk_symbols = ['Γ', 'X', 'W', 'L', 'Γ']

print("Loading Original reconstruction...")
kcoords_orig, eigs_orig, _, _ = load_from_eigdat('eig.dat', 'lat.dat')

print("Loading Band-RI reconstruction...")
kcoords_ri, eigs_ri, _, _ = load_from_eigdat('eig_ri.dat', 'lat.dat')

# Align to VBM
vbm_idx = 3
eigs_dft_shifted  = eigs_dft  - eigs_dft[:, vbm_idx].max()
eigs_orig_shifted = eigs_orig - eigs_orig[:, vbm_idx].max()
eigs_ri_shifted   = eigs_ri   - eigs_ri[:, vbm_idx].max()

x_max = kcoords_dft[-1]
kcoords_orig_scaled = kcoords_orig * (x_max / kcoords_orig[-1])
kcoords_ri_scaled   = kcoords_ri   * (x_max / kcoords_ri[-1])
nbnd_plot = min(nbnd_plot, eigs_orig.shape[1], eigs_ri.shape[1])

# =============================================================================
# Plot
# =============================================================================
print("Creating plot...")
plt.switch_backend('agg')

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), facecolor='none')
ax.set_facecolor('none')

ax.set_xlim(0.0, x_max)
ax.set_ylim(min_plot_energy, max_plot_energy)
ax.set_ylabel('Energy (eV)')
ax.set_xticks(hsk_coords_dft)
ax.set_xticklabels(hsk_symbols)

for hsk in hsk_coords_dft:
    ax.axvline(hsk, color=marp_text_color, linewidth=0.7, alpha=0.5)
ax.axhline(0.0, color=marp_text_color, linestyle='dashed', linewidth=0.7, alpha=0.5)

for i in range(nbnd_plot):
    ax.plot(kcoords_dft,        eigs_dft_shifted[:, i],
            '-',  color=color_pbe,  linewidth=line_width, alpha=alpha,
            label='DFT(QE)'               if i == 0 else None, zorder=3)
    ax.plot(kcoords_orig_scaled, eigs_orig_shifted[:, i],
            '--', color=color_orig, linewidth=line_width, alpha=alpha,
            label='AO-real'  if i == 0 else None, zorder=2)
    ax.scatter(kcoords_ri_scaled, eigs_ri_shifted[:, i],
               color=color_ri, s=6, alpha=alpha,
               label='AO-bands' if i == 0 else None, zorder=4)

ax.legend(loc='lower center', framealpha=legend_alpha, fontsize=0.75*fontsize)
plt.tight_layout()
plt.savefig(output_path, dpi=plot_dpi, transparent=True, bbox_inches='tight')
print(f"Saved to {output_path}")
