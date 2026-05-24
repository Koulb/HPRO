"""
Debug projector normalization in V_nl computation.

Check:
1. Projector self-overlap ⟨β|β⟩
2. Projector-AO overlap ⟨φ|β⟩
3. Compare V_nl computation
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.linalg import eigh

from HPRO.structure import Structure
from HPRO.lcaodata import LCAOData
from HPRO.hrdata import read_hrr
from HPRO.orbutils import OrbPair, read_upf
from HPRO.twocenter import calc_overlap
from HPRO.deephio import load_deeph_HS, get_mat0
from HPRO.constants import hartree2ev


# =============================================================================
# Load structure and data
# =============================================================================
print("=" * 70)
print("Diamond: Projector Normalization Debug")
print("=" * 70)

structure = Structure.from_deeph('./')
aobasis_dir = '../../aobasis'
upf_dir = '../../pseudos'
ecut = 30

# Load AO basis
print("\n[1] Loading AO basis...")
lcaodata = LCAOData(structure, basis_path_root=aobasis_dir, aocode='siesta')
lcaodata.calc_phiQ(ecut * 1.1)
nao = sum(lcaodata.norbfull_spc[spc] for spc in structure.atomic_numbers)
print(f"    Number of AOs: {nao}")

# Load projectors
print("\n[2] Loading projectors...")
funch, funcg, projR = read_hrr(structure, upf_dir, interface='qe')
projR.calc_phiQ(ecut * 1.1)

# Print D_ij from funch
for iatom, spc in enumerate(structure.atomic_numbers):
    D = funch[iatom]
    print(f"    Atom {iatom} (Z={spc}): D_ij shape = {D.shape}")
    print(f"      D_ij diagonal (Hartree): {np.diag(D)}")
    print(f"      D_ij diagonal (eV): {np.diag(D) * hartree2ev}")

# Number of projectors per species
for spc in structure.atomic_species:
    nproj = projR.norbfull_spc[spc]
    print(f"    Species {spc}: {nproj} projector orbitals")

# =============================================================================
# Check projector self-overlap ⟨β|β⟩
# =============================================================================
print("\n" + "=" * 70)
print("[3] Projector self-overlap ⟨β|β⟩")
print("=" * 70)

# Setup orbital pairs for projector self-overlap
orbpairs_proj_self = {}
for ispc in range(structure.nspc):
    spc1 = structure.atomic_species[ispc]
    pairs = []
    for iorb in range(projR.norb_spc[spc1]):
        r1 = projR.phirgrids_spc[spc1][iorb].rcut
        for jorb in range(projR.norb_spc[spc1]):
            r2 = projR.phirgrids_spc[spc1][jorb].rcut
            pair = OrbPair(projR.phiQlist_spc[spc1][iorb],
                          projR.phiQlist_spc[spc1][jorb], r1 + r2, 1)
            pairs.append(pair)
    orbpairs_proj_self[(spc1, spc1)] = pairs

# Compute ⟨β|β⟩ at R=0
R_zero = np.array([[0.0, 0.0, 0.0]])
for spc in structure.atomic_species:
    print(f"\n    Species {spc}:")
    nproj = projR.norb_spc[spc]
    nproj_full = projR.norbfull_spc[spc]
    beta_beta = np.zeros((nproj_full, nproj_full))

    orbpairs = orbpairs_proj_self[(spc, spc)]
    ix = 0
    for iorb in range(nproj):
        slice1 = slice(projR.orbslices_spc[spc][iorb], projR.orbslices_spc[spc][iorb+1])
        for jorb in range(nproj):
            slice2 = slice(projR.orbslices_spc[spc][jorb], projR.orbslices_spc[spc][jorb+1])
            olp = orbpairs[ix].calc(R_zero)
            beta_beta[slice1, slice2] = olp[0]
            ix += 1

    print(f"    ⟨β|β⟩ matrix shape: {beta_beta.shape}")
    print(f"    ⟨β|β⟩ diagonal: {np.diag(beta_beta)}")
    print(f"    Expected for normalized β: 1.0")

    # The relation between D and β normalization:
    # In QE, if β are normalized such that ⟨β_i|β_j⟩ = δ_ij,
    # then V_nl = Σ_ij |β_i⟩ D_ij ⟨β_j|
    # But if β are not normalized, we might need to account for that

    # Check if D * ⟨β|β⟩ gives reasonable values
    D = funch[0]  # Assuming all atoms of same species have same D
    print(f"\n    D × ⟨β|β⟩ diagonal: {np.diag(D @ beta_beta)}")

# =============================================================================
# Check projector-AO overlap ⟨φ|β⟩
# =============================================================================
print("\n" + "=" * 70)
print("[4] Projector-AO overlap ⟨φ|β⟩")
print("=" * 70)

# Setup orbital pairs for projector-AO overlap
orbpairs_proj_ao = {}
for ispc in range(structure.nspc):
    for jspc in range(structure.nspc):
        spc1 = structure.atomic_species[ispc]
        spc2 = structure.atomic_species[jspc]
        pairs = []
        for jorb in range(lcaodata.norb_spc[spc2]):
            r2 = lcaodata.phirgrids_spc[spc2][jorb].rcut
            for iorb in range(projR.norb_spc[spc1]):
                r1 = projR.phirgrids_spc[spc1][iorb].rcut
                pair = OrbPair(projR.phiQlist_spc[spc1][iorb],
                              lcaodata.phiQlist_spc[spc2][jorb], r1 + r2, 1)
                pairs.append(pair)
        orbpairs_proj_ao[(spc1, spc2)] = pairs

# Compute the full overlap matrix
olp_proj_ao = calc_overlap(projR, orbpairs_proj_ao, lcaodata, Ecut=ecut)

print(f"\n    olp_proj_ao contains {olp_proj_ao.npairs} blocks")

# Find the on-site block (R=0, same atom)
for ipair in range(olp_proj_ao.npairs):
    trans = olp_proj_ao.translations[ipair]
    atm1, atm2 = olp_proj_ao.atom_pairs[ipair]
    if np.allclose(trans, [0, 0, 0]) and atm1 == atm2:
        mat = olp_proj_ao.mats[ipair]
        print(f"\n    On-site block (atom {atm1}):")
        print(f"    Shape: {mat.shape} (n_proj × n_ao)")
        print(f"    Frobenius norm: {np.linalg.norm(mat, 'fro'):.4f}")
        print(f"    Max abs value: {np.max(np.abs(mat)):.4f}")
        print(f"    First few diagonal values: {np.diag(mat)[:5]}")
        break

# =============================================================================
# Compute V_nl and compare
# =============================================================================
print("\n" + "=" * 70)
print("[5] V_nl computation")
print("=" * 70)

# Compute V_nl using get_mat0
trans, atoms, mats0 = get_mat0(olp_proj_ao, funch)
print(f"\n    get_mat0 returned {len(mats0)} matrix blocks")

# Find on-site V_nl at (R=0, same atom)
for i, (t, a) in enumerate(zip(trans, atoms)):
    if np.allclose(t, [0, 0, 0]) and a[0] == a[1]:
        print(f"\n    On-site V_nl block (atoms {a[0]}-{a[1]}):")
        print(f"    Shape: {mats0[i].shape}")
        print(f"    Diagonal (first 8, eV): {np.diag(mats0[i])[:8] * hartree2ev}")
        break

# Load H from file for comparison
matH = load_deeph_HS('./', 'hamiltonians.h5', energy_unit=True)
matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)

k_gamma = np.array([0.0, 0.0, 0.0])
Hk = matH.r2k(k_gamma).toarray()
Sk = matS.r2k(k_gamma).toarray()

# Compute kinetic energy for comparison
orbpairs_kin = {}
for ispc in range(structure.nspc):
    for jspc in range(structure.nspc):
        spc1 = structure.atomic_species[ispc]
        spc2 = structure.atomic_species[jspc]
        pairs = []
        for jorb in range(lcaodata.norb_spc[spc2]):
            r2 = lcaodata.phirgrids_spc[spc2][jorb].rcut
            for iorb in range(lcaodata.norb_spc[spc1]):
                r1 = lcaodata.phirgrids_spc[spc1][iorb].rcut
                pair = OrbPair(lcaodata.phiQlist_spc[spc1][iorb],
                              lcaodata.phiQlist_spc[spc2][jorb], r1 + r2, 2)
                pairs.append(pair)
        orbpairs_kin[(spc1, spc2)] = pairs

Hkin_mat = calc_overlap(lcaodata, orbpairs_kin, Ecut=ecut)
Hkin_k = Hkin_mat.r2k(k_gamma).toarray()

print(f"\n    ||T||_F:    {np.linalg.norm(Hkin_k, 'fro'):.4f}")
print(f"    ||H_file||_F: {np.linalg.norm(Hk, 'fro'):.4f}")
print(f"    ||S||_F:    {np.linalg.norm(Sk, 'fro'):.4f}")

# =============================================================================
# Check the formula: V_nl = ⟨φ|β⟩.T @ D @ ⟨β|φ⟩
# =============================================================================
print("\n" + "=" * 70)
print("[6] Direct V_nl computation check")
print("=" * 70)

# Get on-site projector-AO overlap for atom 0
for ipair in range(olp_proj_ao.npairs):
    trans = olp_proj_ao.translations[ipair]
    atm1, atm2 = olp_proj_ao.atom_pairs[ipair]
    if np.allclose(trans, [0, 0, 0]) and atm1 == 0 and atm2 == 0:
        # mat shape: (n_proj, n_ao)
        mat = olp_proj_ao.mats[ipair]
        D = funch[0]

        # V_nl = mat.T @ D @ mat (on-site contribution)
        # mat.T: (n_ao, n_proj)
        # D: (n_proj, n_proj)
        # mat: (n_proj, n_ao)
        Vnl_onsite = mat.T @ D @ mat

        print(f"\n    On-site V_nl for atom 0:")
        print(f"    mat shape: {mat.shape}")
        print(f"    D shape: {D.shape}")
        print(f"    V_nl shape: {Vnl_onsite.shape}")
        print(f"    V_nl diagonal (first 8, eV): {np.diag(Vnl_onsite)[:8] * hartree2ev}")

        # Compare with the value from get_mat0
        # (which should be the same for on-site same-atom block)
        break

# =============================================================================
# Read UPF directly to check raw values
# =============================================================================
print("\n" + "=" * 70)
print("[7] Raw UPF file check")
print("=" * 70)

upf_path = f'{upf_dir}/C.upf'
funch_upf, projR_list = read_upf(upf_path)
print(f"\n    D_ij from UPF (shape {funch_upf.shape}):")
print(f"    Diagonal (Hartree): {np.diag(funch_upf)}")
print(f"    Diagonal (eV): {np.diag(funch_upf) * hartree2ev}")

# Check projector radial functions
print(f"\n    Projector functions:")
for i, proj in enumerate(projR_list):
    print(f"    Projector {i}: l={proj.l}, rcut={proj.rcut:.4f}")
    # Check normalization: ∫ |β(r)|² r² dr
    r = proj.rgd.rfunc
    dr = np.gradient(r)
    norm = np.sum(proj.func**2 * r**2 * dr)
    print(f"      ∫ |β(r)|² r² dr = {norm:.6f}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
