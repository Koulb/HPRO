from HPRO.lcaodiag import LCAODiagKernel
from HPRO.deephio import load_deeph_HS
from HPRO.constants import hartree2ev
import numpy as np
from scipy.linalg import eigh
import os

# FCC Diamond path: Γ → X → W → L → Γ
kpath = [[0.000,  0.000,  0.000],
         [0.500,  0.000,  0.500],
         [0.500,  0.250,  0.750],
         [0.500,  0.500,  0.500],
         [0.000,  0.000,  0.000]]
kpath_npts = [20, 15, 15, 20, 1]
kpath_labels = ['Γ', 'X', 'W', 'L', 'Γ']
nbnd = 20

# Original reconstruction
kernel = LCAODiagKernel()
kernel.setk(kpath, kpath_npts, kpath_labels)
kernel.load_deeph_mats('./')
kernel.diag(nbnd=nbnd, efermi=None)
kernel.write('./')

# Band-RI reconstruction — load directly to skip hermitianize
if os.path.exists('hamiltonians_ri.h5'):
    print('\n--- Band-RI reconstruction (direct loading, no hermitianize) ---')
    matH_ri = load_deeph_HS('./', 'hamiltonians_ri.h5', energy_unit=True)
    matS = load_deeph_HS('./', 'overlaps.h5', energy_unit=False)
    matS.hermitianize()
    nao = matH_ri.norb_total

    # Generate band-path k-points
    kpts_path = np.array(kpath)
    npts = np.array(kpath_npts)
    kpts_full = []
    hsk_positions = []
    hsk_symbols_out = []
    for iseg in range(len(kpts_path) - 1):
        wk = npts[iseg]
        kstart = kpts_path[iseg]
        kend = kpts_path[iseg + 1]
        for i in range(wk):
            t = i / wk
            kpts_full.append(kstart + t * (kend - kstart))
            if i == 0:
                hsk_positions.append(len(kpts_full) - 1)
                hsk_symbols_out.append(kpath_labels[iseg])
    kpts_full.append(kpts_path[-1])
    hsk_positions.append(len(kpts_full) - 1)
    hsk_symbols_out.append(kpath_labels[-1])
    kpts_full = np.array(kpts_full)
    nk = len(kpts_full)

    # Diagonalize at each k-point: hermitianize H(k) only (not H(R))
    eigs_all = np.zeros((nk, nbnd))
    for ik in range(nk):
        kpt = kpts_full[ik]
        Hk = matH_ri.r2k(kpt).toarray()
        Sk = matS.r2k(kpt).toarray()
        Hk = 0.5 * (Hk + Hk.conj().T)
        Sk = 0.5 * (Sk + Sk.conj().T)
        eigs_k, _ = eigh(Hk, Sk)
        eigs_all[ik, :nbnd] = eigs_k[:nbnd]

    # Write eig_ri.dat
    with open('eig_ri.dat', 'w') as f:
        f.write('Band energies in eV\n')
        f.write('      nk    nbnd\n')
        f.write(f'      {nk:2d}      {nbnd:2d}\n')
        for ik in range(nk):
            kpt = kpts_full[ik]
            if ik in hsk_positions:
                idx = hsk_positions.index(ik)
                f.write(f'  {kpt[0]:.9f}  {kpt[1]:.9f}  {kpt[2]:.9f}      {nbnd:2d}  {hsk_symbols_out[idx]}\n')
            else:
                f.write(f'  {kpt[0]:.9f}  {kpt[1]:.9f}  {kpt[2]:.9f}      {nbnd:2d}\n')
            for ibnd in range(nbnd):
                eig_eV = eigs_all[ik, ibnd] * hartree2ev
                f.write(f'       1  {ibnd+1:5d} {eig_eV:15.9f}\n')
    print(f'  Wrote eig_ri.dat: {nk} k-points, {nbnd} bands')
