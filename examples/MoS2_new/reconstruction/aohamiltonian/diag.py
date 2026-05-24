from HPRO.lcaodiag import LCAODiagKernel
import os

kpath = [[0.000000000000,  0.000000000000,  0.000000000000],
         [0.333333333333,  0.333333333333,  0.000000000000],
         [0.500000000000,  0.000000000000,  0.000000000000],
         [0.000000000000,  0.000000000000,  0.000000000000]]
kpath_npts = [20, 10, 17, 1]
kpath_labels = ['Γ', 'K', 'M', 'Γ']

# Original reconstruction
kernel = LCAODiagKernel()
kernel.setk(kpath, kpath_npts, kpath_labels)
kernel.load_deeph_mats('./')
kernel.diag(nbnd=72, efermi=None)
kernel.write('./')

zz
