"""
HSE Band-RI reconstruction convergence for Diamond.

Builds H_bandRI(R) from HSE eigenvalues and projections at nscf k-points,
then FTs back to k-space and diagonalizes. Compares Band-RI eigenvalues
against QE HSE reference as a function of nbands used.

H_ij(R) = Σ_{nk} w_k A†_{i,0,nk} ε_{nk} A_{j,R,nk}
where A_{j,R,nk} = ⟨ψ_{nk}|φ_j(R)⟩ = A_{j,0,nk} * exp(-2πi k·R)
"""

import numpy as np
import h5py
import xml.etree.ElementTree as ET
from scipy.io import FortranFile
from scipy.linalg import eigh
import matplotlib.pyplot as plt

from HPRO.deephio import load_deeph_HS, get_Us_openmx2wiki
from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData, calc_FT_kg_orb_spcs
from HPRO.constants import hartree2ev


def parse_all_kpoints_xml(xml_path):
    """Parse all k-points and eigenvalues from QE XML file."""
    tree = ET.parse(xml_path)
    kpoints = []
    eigenvalues = []
    weights = []
    for ks_energies in tree.iter('ks_energies'):
        k_point_elem = ks_energies.find('k_point')
        kpt = np.array([float(x) for x in k_point_elem.text.split()])
        kpoints.append(kpt)
        weight = float(k_point_elem.get('weight', 1.0))
        weights.append(weight)
        eigs = np.array([float(x) for x in ks_energies.find('eigenvalues').text.split()])
        eigenvalues.append(eigs)
    return kpoints, eigenvalues, np.array(weights)


def read_kpoints_from_pwinput(pw_path):
    """Read k-points from QE pw.in file (crystal coordinates)."""
    kpoints = []
    weights = []
    with open(pw_path, 'r') as f:
        lines = f.readlines()
    in_kpoints = False
    nkpt = 0
    kpt_count = 0
    for line in lines:
        line = line.strip()
        if line.upper().startswith('K_POINTS'):
            in_kpoints = True
            continue
        if in_kpoints and nkpt == 0:
            nkpt = int(line)
            continue
        if in_kpoints and kpt_count < nkpt:
            parts = line.split()
            kpt = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            weight = float(parts[3]) if len(parts) > 3 else 1.0 / nkpt
            kpoints.append(kpt)
            weights.append(weight)
            kpt_count += 1
    return kpoints, np.array(weights)


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
    """Compute AO functions φ_μ(k+G) in plane-wave basis."""
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
    """Compute overlap matrix A[n,μ] = ⟨ψ_n|φ_μ⟩ in PW basis."""
    nbnd = len(psi_list)
    nao = phi_kg.shape[1]
    norm_factor = 1.0 / np.sqrt(cell_volume)
    A = np.zeros((nbnd, nao), dtype=np.complex128)
    for n, psi in enumerate(psi_list):
        A[n, :] = norm_factor * np.conj(psi) @ phi_kg
    return A


def diagonalize_generalized(H, S):
    """Solve generalized eigenvalue problem: H c = ε S c."""
    H_herm = 0.5 * (H + H.conj().T)
    S_herm = 0.5 * (S + S.conj().T)
    eigenvalues, _ = eigh(H_herm, S_herm)
    return eigenvalues


def r_in_grid_range(R, grid_dims):
    """Check if R vector is within the k-grid's fundamental domain."""
    for i in range(3):
        N = grid_dims[i]
        if N == 1:
            if R[i] != 0:
                return False
        else:
            lo = -(N // 2)
            hi = (N - 1) // 2
            if R[i] < lo or R[i] > hi:
                return False
    return True


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("Diamond HSE: Band-RI Reconstruction Convergence")
print("=" * 70)

# Paths
nscf_save_dir = '../../nscf/diamond.save'
xml_path = f'{nscf_save_dir}/data-file-schema.xml'
pw_input_path = '../../nscf/pw.in'
aobasis_dir = '../../aobasis'
ecut = 30
max_nbands = 100

# Load structure
print("\n[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))
print(f"    Cell volume: {cell_volume:.4f} bohr^3")

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load S from file (functional-independent)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get k-points and HSE eigenvalues
kpoints, kweights = read_kpoints_from_pwinput(pw_input_path)
_, all_eigenvalues, _ = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# Normalize weights
kweights_norm = kweights / np.sum(kweights)

# =============================================================================
# Load wavefunctions and compute A_0(k) for all k-points
# =============================================================================
print("\n[2] Loading HSE wavefunctions and computing A_0(k)...")

A_0_list = []
valid_kpts = []
valid_eigs = []
valid_weights = []

for ik in range(nkpt):
    kpt = kpoints[ik]
    wfc_path = f'{nscf_save_dir}/wfc{ik+1}.dat'
    try:
        psi_list, miller = read_wfc_qe(wfc_path, max_nbands)
        phi_kg = compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut)
        A_k = compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume)
        A_0_list.append(A_k)
        valid_kpts.append(kpt)
        valid_eigs.append(all_eigenvalues[ik])
        valid_weights.append(kweights[ik])
    except Exception as e:
        print(f"    Warning: Could not load k-point {ik+1}: {e}")
        continue

valid_weights = np.array(valid_weights)
valid_weights_norm = valid_weights / np.sum(valid_weights)
print(f"    Successfully loaded {len(valid_kpts)} k-points")

# =============================================================================
# Determine k-grid dimensions
# =============================================================================
kpts_arr = np.array(valid_kpts)
grid_dims = []
for i in range(3):
    unique_vals = np.unique(np.round(kpts_arr[:, i], decimals=8))
    grid_dims.append(len(unique_vals))
grid_dims = np.array(grid_dims)
print(f"    K-grid dimensions: {grid_dims[0]}x{grid_dims[1]}x{grid_dims[2]}")

# Generate canonical R grid
canonical_R_set = set()
for r1 in range(-(grid_dims[0]//2), (grid_dims[0]+1)//2):
    for r2 in range(-(grid_dims[1]//2), (grid_dims[1]+1)//2):
        for r3 in range(-(grid_dims[2]//2), (grid_dims[2]+1)//2):
            canonical_R_set.add((r1, r2, r3))
all_R = sorted(canonical_R_set)
print(f"    Canonical R set: {len(canonical_R_set)} vectors")

# =============================================================================
# Band-RI convergence: eigenvalue MAE vs nbands
# =============================================================================
print("\n" + "=" * 70)
print("[3] Band-RI eigenvalue convergence vs nbands")
print("=" * 70)

band_counts = [10, 20, 30, 50, 80, 100]
target_bands = [4, 8]

# Get Fermi energy for reference
tree = ET.parse(xml_path)
band_elem = tree.find('.//output/band_structure')
efermi_ha = float(band_elem.find('fermi_energy').text)
print(f"    Fermi energy: {efermi_ha * hartree2ev:.4f} eV")

results = {nb: {tb: [] for tb in target_bands} for nb in band_counts}

for nbands in band_counts:
    print(f"\n  --- nbands = {nbands} ---")

    # Build H_bandRI(R)
    H_bandri_R = {}
    for R in all_R:
        R_vec = np.array([int(R[0]), int(R[1]), int(R[2])], dtype=float)
        H_R = np.zeros((nao, nao), dtype=np.complex128)
        for ik in range(len(valid_kpts)):
            kpt = valid_kpts[ik]
            nb_use = min(nbands, len(valid_eigs[ik]))
            eigs_k = valid_eigs[ik][:nb_use]
            A_0_k = A_0_list[ik][:nb_use]
            phase = np.exp(-2j * np.pi * np.dot(kpt, R_vec))
            A_R_k = A_0_k * phase
            eA_R = eigs_k[:, None] * A_R_k
            H_R += valid_weights_norm[ik] * (A_0_k.conj().T @ eA_R)
        H_bandri_R[R] = H_R

    # Diagonalize at each nscf k-point using FT: H(k) = Σ_R H(R) exp(2πi k·R)
    mae_per_target = {tb: [] for tb in target_bands}
    for ik in range(len(valid_kpts)):
        kpt = valid_kpts[ik]
        eigs_ref = valid_eigs[ik]

        # Build H(k) from H(R)
        Hk = np.zeros((nao, nao), dtype=np.complex128)
        for R in all_R:
            R_vec = np.array([int(R[0]), int(R[1]), int(R[2])], dtype=float)
            phase = np.exp(2j * np.pi * np.dot(kpt, R_vec))
            Hk += H_bandri_R[R] * phase

        # Build S(k)
        Sk = matS.r2k(kpt).toarray()

        # Hermitianize at k-space level
        Hk = 0.5 * (Hk + Hk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)

        eigs_ri, _ = eigh(Hk, Sk)

        for tb in target_bands:
            mae = np.mean(np.abs(eigs_ri[:tb] - eigs_ref[:tb])) * hartree2ev * 1000
            mae_per_target[tb].append(mae)

    for tb in target_bands:
        avg_mae = np.mean(mae_per_target[tb])
        results[nbands][tb] = avg_mae
        print(f"    MAE (lowest {tb} bands): {avg_mae:.2f} meV")

    # Save hamiltonians_ri.h5 for the max nbands case
    if nbands == max_nbands:
        print(f"\n    Saving hamiltonians_ri.h5 for nbands={nbands}...")
        lcaodata_deeph = LCAOData(structure, None, basis_path_root='./', aocode='deeph')
        Us = get_Us_openmx2wiki(lcaodata_deeph.ls_spc)
        norb_per_atom = [lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers]
        orb_cumsum = np.concatenate(([0], np.cumsum(norb_per_atom)))
        natoms = len(structure.atomic_numbers)
        with h5py.File('hamiltonians_ri.h5', 'w') as f_out:
            nkeys = 0
            for R in sorted(canonical_R_set):
                for iatom in range(natoms):
                    for jatom in range(natoms):
                        spc1 = structure.atomic_numbers[iatom]
                        spc2 = structure.atomic_numbers[jatom]
                        i0, i1 = orb_cumsum[iatom], orb_cumsum[iatom + 1]
                        j0, j1 = orb_cumsum[jatom], orb_cumsum[jatom + 1]
                        block_wiki = H_bandri_R[R][i0:i1, j0:j1].real
                        block_openmx = Us[spc1].T @ block_wiki @ Us[spc2]
                        block_eV = block_openmx * hartree2ev
                        key_str = str([R[0], R[1], R[2], iatom + 1, jatom + 1])
                        f_out[key_str] = block_eV
                        nkeys += 1
        print(f"    Saved {nkeys} blocks for {len(canonical_R_set)} canonical R-vectors")

# =============================================================================
# Convergence table
# =============================================================================
print("\n" + "=" * 70)
print("[4] Convergence Table: MAE (meV) vs nbands")
print("=" * 70)

header = f"{'nbands':>8}"
for tb in target_bands:
    header += f" {'MAE ' + str(tb) + ' bands':>16}"
print(header)
print("-" * (8 + 16 * len(target_bands)))

for nbands in band_counts:
    row = f"{nbands:8d}"
    for tb in target_bands:
        row += f" {results[nbands][tb]:16.2f}"
    print(row)

# =============================================================================
# Convergence plot
# =============================================================================
print("\n[5] Creating convergence plot...")
plt.switch_backend('agg')
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for tb in target_bands:
    maes = [results[nb][tb] for nb in band_counts]
    ax.plot(band_counts, maes, 'o-', linewidth=2, markersize=8, label=f'Lowest {tb} bands')

ax.set_xlabel('Number of bands (nbands)', fontsize=13)
ax.set_ylabel('MAE vs QE HSE (meV)', fontsize=13)
ax.set_title('Band-RI convergence for HSE (Diamond, 6x6x6 k-grid)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('convergence_hse.png', dpi=300)
print("    Saved convergence_hse.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Diamond HSE Band-RI reconstruction convergence:

Eigenvalue MAE (meV) vs QE HSE reference at {len(valid_kpts)} nscf k-points:
""")
for nb in band_counts:
    line = f"  nbands={nb:4d}:"
    for tb in target_bands:
        line += f"  {tb} bands = {results[nb][tb]:.2f} meV"
    print(line)

print(f"""
hamiltonians_ri.h5 saved for nbands={max_nbands}
Convergence plot: convergence_hse.png
""")
print("=" * 70)
print("Done!")
print("=" * 70)
