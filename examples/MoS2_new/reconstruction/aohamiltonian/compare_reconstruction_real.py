"""
Direct real-space Band-RI reconstruction for MoS2.

H_ij(R) = Σ_{nk} w_k A†_{i,0,nk} ε_{nk} A_{j,R,nk}

where A_{j,R,nk} = ⟨ψ_{nk}|φ_j(R)⟩ = A_{j,0,nk} * exp(-2πi k·R)

Compares:
1. H(R) from direct real-space Band-RI
2. H(R) from original reconstruction in hamiltonians.h5

Saves result to hamiltonians_ri.h5.
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


def compute_ao_in_pw_basis(miller, gprim, kpt, structure, lcaodata, ecut, R_cart=None):
    """Compute AO functions φ_μ^R(k+G) in plane-wave basis.

    When R_cart is provided, computes AOs translated by lattice vector R:
        φ_μ^R(k+G) = FT_μ(k+G) * exp(-i(k+G)·(τ_a + R))

    Since (k+G)·R = 2π(k_cryst + g_cryst)·R_cryst and g·R is integer,
    the R-dependent phase is exp(-2πi k·R), uniform across all G vectors.
    """
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
        if R_cart is not None:
            tau = tau + R_cart
        phase = np.exp(-1j * kgcart @ tau)
        phi_kg[:, iao:iao+nao_atom] = FT_kg_orb_spcs[spc] * phase[:, None]
        iao += nao_atom
    return phi_kg


def compute_overlap_matrix_pw(psi_list, phi_kg, cell_volume):
    """Compute overlap matrix A[n,μ] = ⟨ψ_n|φ_μ⟩ in PW basis.

    Returns A with shape (nbnd, nao). The normalization factor
    1/√Ω accounts for the plane-wave basis normalization.
    """
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


# =============================================================================
# Main
# =============================================================================
print("=" * 70)
print("MoS2: Direct Real-Space Band-RI Reconstruction")
print("=" * 70)

# Paths
nscf_save_dir = '../../nscf/MoS2.save'
xml_path = f'{nscf_save_dir}/data-file-schema.xml'
pw_input_path = '../../nscf/pw.in'
aobasis_dir = '../../aobasis'
ecut = 30
max_nbands = 400

# Load structure
print("\n[1] Loading structure and data...")
rprim, gprim = get_structure_from_xml(xml_path)
cell_volume = np.abs(np.linalg.det(rprim))
print(f"    Cell volume: {cell_volume:.4f} bohr³")

structure = Structure.from_deeph('./')
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load H and S from file
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

# Get all H(R) from original reconstruction
matscsr_R = matH.to_csr()
R_vectors = list(matscsr_R.keys())
print(f"    Number of R-vectors: {len(R_vectors)}")

H_R0_file = matscsr_R[(0, 0, 0)].toarray()
print(f"    ||H(R=0) from file||_F: {np.linalg.norm(H_R0_file, 'fro'):.4f}")

# Get k-points
kpoints, kweights = read_kpoints_from_pwinput(pw_input_path)
_, all_eigenvalues, _ = parse_all_kpoints_xml(xml_path)
nkpt = len(kpoints)
print(f"    Number of k-points: {nkpt}")

# Normalize weights
kweights_norm = kweights / np.sum(kweights)

# =============================================================================
# Load wavefunctions and compute A_0(k) for all k-points
# =============================================================================
print("\n[2] Loading wavefunctions and computing A_0(k)...")

A_0_list = []     # A_0(k) for each k-point
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
# Compute H_bandRI(R) directly using real-space formula
# H(R) = Σ_k w_k A†_0(k) diag(ε_k) A_R(k)
# where A_R(k) = A_0(k) * exp(-2πi k·R)
# =============================================================================
print("\n[3] Computing H_bandRI(R) via direct real-space formula...")
print("    H(R) = Σ_k w_k A†_{0,k} diag(ε_k) A_{R,k}")
print(f"    Using {max_nbands} bands")

H_bandri_R = {}
for R in R_vectors:
    R_vec = np.array([int(R[0]), int(R[1]), int(R[2])], dtype=float)
    H_R = np.zeros((nao, nao), dtype=np.complex128)
    for ik in range(len(valid_kpts)):
        kpt = valid_kpts[ik]
        eigs_k = valid_eigs[ik][:max_nbands]
        A_0_k = A_0_list[ik]

        # A_R(k) = A_0(k) * exp(-2πi k·R)
        phase = np.exp(-2j * np.pi * np.dot(kpt, R_vec))
        A_R_k = A_0_k * phase

        # H(R) += w_k * A†_0(k) diag(ε_k) A_R(k)
        # = w_k * (A_0^H diag(ε) A_0) * phase
        # Using matmul: A_0^H @ (ε[:,None] * A_R)
        eA_R = eigs_k[:, None] * A_R_k   # diag(ε) @ A_R
        H_R += valid_weights_norm[ik] * (A_0_k.conj().T @ eA_R)

    H_bandri_R[R] = H_R

print(f"    Computed H(R) for {len(R_vectors)} R-vectors")

# Check imaginary part
max_imag = max(np.max(np.abs(H_bandri_R[R].imag)) for R in R_vectors)
max_real = max(np.max(np.abs(H_bandri_R[R].real)) for R in R_vectors)
print(f"    Max |imag(H(R))|: {max_imag:.2e}")
print(f"    Max |real(H(R))|: {max_real:.2e}")
print(f"    Ratio imag/real:  {max_imag/max_real:.2e}")

# =============================================================================
# Test 1: H(R) comparison at all R vectors
# =============================================================================
print("\n" + "=" * 70)
print("[4] H(R) comparison: Band-RI (real-space) vs Original")
print("=" * 70)

errors_by_R = {}
print(f"\n{'R vector':>20} {'||H_orig(R)||':>14} {'||H_bandRI(R)||':>16} {'Rel. Error':>12}")
print("-" * 65)

R_sorted = sorted(R_vectors, key=lambda R: np.linalg.norm(matscsr_R[R].toarray(), 'fro'), reverse=True)

for R in R_sorted[:15]:
    R_vec = np.array([int(R[0]), int(R[1]), int(R[2])])
    H_orig_R = matscsr_R[R].toarray()
    norm_orig = np.linalg.norm(H_orig_R, 'fro')
    H_ri = H_bandri_R[R].real
    norm_bandri = np.linalg.norm(H_ri, 'fro')

    if norm_orig > 1e-10:
        rel_error = np.linalg.norm(H_ri - H_orig_R, 'fro') / norm_orig
    else:
        rel_error = 0.0
    errors_by_R[R] = rel_error

    R_str = f"({R[0]:2d},{R[1]:2d},{R[2]:2d})"
    print(f"{R_str:>20} {norm_orig:14.6f} {norm_bandri:16.6f} {rel_error*100:11.2f}%")

if len(R_vectors) > 15:
    print(f"    ... and {len(R_vectors) - 15} more R-vectors")

# =============================================================================
# Test 2: H(R=0) convergence with nbands
# =============================================================================
print("\n" + "=" * 70)
print("[5] H(R=0) convergence: Band-RI vs Original")
print("=" * 70)

band_counts = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400]
errors_vs_R0 = []

print(f"{'nbands':>8} {'||H_ksum - H(R=0)||/||H(R=0)||':>35}")
print("-" * 45)

for nbands in band_counts:
    H_loc = np.zeros((nao, nao), dtype=np.complex128)
    for ik in range(len(valid_kpts)):
        eigs_k = valid_eigs[ik][:nbands]
        A_0_k = A_0_list[ik][:nbands]
        # R=0: phase = 1, so A_R = A_0
        eA = eigs_k[:, None] * A_0_k
        H_loc += valid_weights_norm[ik] * (A_0_k.conj().T @ eA)

    err = np.linalg.norm(H_loc.real - H_R0_file, 'fro') / np.linalg.norm(H_R0_file, 'fro')
    errors_vs_R0.append(err)
    print(f"{nbands:8d} {err*100:34.2f}%")

print("-" * 45)

# =============================================================================
# Test 3: Eigenvalue comparison for lowest bands
# =============================================================================
print("\n" + "=" * 70)
print("[6] Eigenvalue comparison at nscf k-points")
print("=" * 70)

target_bands = [4, 8, 16]
tree = ET.parse(xml_path)
band_elem = tree.find('.//output/band_structure')
efermi_ha = float(band_elem.find('fermi_energy').text)
print(f"    Fermi energy: {efermi_ha * hartree2ev:.4f} eV")

mae_original = {n: [] for n in target_bands}
mae_bandri = {n: [] for n in target_bands}

for ik in range(len(valid_kpts)):
    kpt = valid_kpts[ik]
    eigs_ref = valid_eigs[ik]

    # Original reconstruction
    Hk_orig = matH.r2k(kpt).toarray()
    Sk = matS.r2k(kpt).toarray()
    eigs_orig = diagonalize_generalized(Hk_orig, Sk)

    # Band-RI: H(k) = Σ_R H(R) exp(2πi k·R)
    H_bandri_k = np.zeros((nao, nao), dtype=np.complex128)
    for R in R_vectors:
        R_vec = np.array([int(R[0]), int(R[1]), int(R[2])], dtype=float)
        phase = np.exp(2j * np.pi * np.dot(kpt, R_vec))
        H_bandri_k += H_bandri_R[R] * phase
    eigs_bandri = diagonalize_generalized(H_bandri_k, Sk)

    for n in target_bands:
        mae_o = np.mean(np.abs(eigs_orig[:n] - eigs_ref[:n])) * hartree2ev * 1000
        mae_b = np.mean(np.abs(eigs_bandri[:n] - eigs_ref[:n])) * hartree2ev * 1000
        mae_original[n].append(mae_o)
        mae_bandri[n].append(mae_b)

print("\nMAE vs QE reference (averaged over k-points):")
print(f"{'Target bands':>14} {'Original (meV)':>16} {'Band-RI (meV)':>16}")
print("-" * 50)
for n in target_bands:
    avg_orig = np.mean(mae_original[n])
    avg_bandri = np.mean(mae_bandri[n])
    print(f"{n:14d} {avg_orig:16.1f} {avg_bandri:16.1f}")

# =============================================================================
# Save H_bandRI(R) to hamiltonians_ri.h5
# =============================================================================
print("\n" + "=" * 70)
print("[7] Saving to hamiltonians_ri.h5")
print("=" * 70)

# Get orbital rotation matrices (wiki <-> openmx)
lcaodata_deeph = LCAOData(structure, None, basis_path_root='./', aocode='deeph')
Us = get_Us_openmx2wiki(lcaodata_deeph.ls_spc)

# Orbital slices per atom
norb_per_atom = [lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers]
orb_cumsum = np.concatenate(([0], np.cumsum(norb_per_atom)))

# Read original h5 keys to reproduce the same structure
with h5py.File('hamiltonians.h5', 'r') as f_orig:
    orig_keys = list(f_orig.keys())

with h5py.File('hamiltonians_ri.h5', 'w') as f_out:
    for key_str in orig_keys:
        Rijab = eval(key_str)
        R = tuple(Rijab[:3])
        iatom = Rijab[3] - 1  # 0-based
        jatom = Rijab[4] - 1

        spc1 = structure.atomic_numbers[iatom]
        spc2 = structure.atomic_numbers[jatom]

        # Extract block from full H(R) (wiki convention, Hartree)
        i0, i1 = orb_cumsum[iatom], orb_cumsum[iatom + 1]
        j0, j1 = orb_cumsum[jatom], orb_cumsum[jatom + 1]
        block_wiki = H_bandri_R[R][i0:i1, j0:j1].real

        # wiki -> openmx rotation, Hartree -> eV
        block_openmx = Us[spc1].T @ block_wiki @ Us[spc2]
        block_eV = block_openmx * hartree2ev

        f_out[key_str] = block_eV

print(f"    Saved {len(orig_keys)} blocks to hamiltonians_ri.h5")

# Verify roundtrip: reload and compare
matH_ri = load_deeph_HS('./', 'hamiltonians_ri.h5', energy_unit=True)
matscsr_R_ri = matH_ri.to_csr()
H_R0_ri = matscsr_R_ri[(0, 0, 0)].toarray()
roundtrip_err = np.linalg.norm(H_R0_ri - H_bandri_R[(0, 0, 0)].real, 'fro')
print(f"    Roundtrip error at R=0: {roundtrip_err:.2e}")

# =============================================================================
# Create plots
# =============================================================================
print("\n" + "=" * 70)
print("[8] Creating plots...")
print("=" * 70)

plt.switch_backend('agg')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: H(R=0) convergence with nbands
ax1 = axes[0]
ax1.plot(band_counts, [e * 100 for e in errors_vs_R0], 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of bands', fontsize=12)
ax1.set_ylabel('Relative error (%)', fontsize=12)
ax1.set_title('H(R=0): Band-RI vs Original', fontsize=13)
ax1.grid(True, alpha=0.3)

# Plot 2: H(R) errors
ax2 = axes[1]
R_labels = []
R_errors = []
for R in R_sorted:
    if R in errors_by_R:
        R_labels.append(f"({R[0]},{R[1]},{R[2]})")
        R_errors.append(errors_by_R[R] * 100)
ax2.bar(range(len(R_errors[:12])), R_errors[:12], color='steelblue', edgecolor='navy')
ax2.set_xticks(range(len(R_labels[:12])))
ax2.set_xticklabels(R_labels[:12], rotation=45, ha='right', fontsize=9)
ax2.set_xlabel('R vector', fontsize=12)
ax2.set_ylabel('Relative error (%)', fontsize=12)
ax2.set_title('H(R): Band-RI vs Original', fontsize=13)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Eigenvalue MAE
ax3 = axes[2]
x = np.arange(len(target_bands))
width = 0.35
ax3.bar(x - width/2, [np.mean(mae_original[n]) for n in target_bands], width, label='Original', color='coral')
ax3.bar(x + width/2, [np.mean(mae_bandri[n]) for n in target_bands], width, label='Band-RI', color='seagreen')
ax3.set_xlabel('Number of lowest bands', fontsize=12)
ax3.set_ylabel('MAE vs QE (meV)', fontsize=12)
ax3.set_title('Eigenvalue Comparison', fontsize=13)
ax3.set_xticks(x)
ax3.set_xticklabels([str(n) for n in target_bands])
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('comparison_real.png', dpi=300)
print("    Saved comparison_real.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
MoS2 direct real-space Band-RI reconstruction:

1. H(R=0) convergence (Band-RI vs Original):
   - 20 bands: {errors_vs_R0[1]*100:.1f}% error
   - 50 bands: {errors_vs_R0[4]*100:.1f}% error
   - 400 bands: {errors_vs_R0[-1]*100:.1f}% error

2. H(R) comparison at finite R (with {max_nbands} bands):""")

for i, R in enumerate(R_sorted[:5]):
    R_str = f"({R[0]},{R[1]},{R[2]})"
    norm_orig = np.linalg.norm(matscsr_R[R].toarray(), 'fro')
    err = errors_by_R.get(R, 0) * 100
    print(f"   R={R_str:>12}: ||H_orig||={norm_orig:.4f}, error={err:.1f}%")

print(f"""
3. Eigenvalue MAE vs QE reference (averaged over {len(valid_kpts)} k-points):""")
for n in target_bands:
    print(f"   Lowest {n} bands: Original = {np.mean(mae_original[n]):.1f} meV, Band-RI = {np.mean(mae_bandri[n]):.1f} meV")

print(f"""
4. Saved hamiltonians_ri.h5 ({len(orig_keys)} blocks, {len(R_vectors)} R-vectors)
   Roundtrip error: {roundtrip_err:.2e}
""")
print("=" * 70)
print("Done!")
print("=" * 70)
