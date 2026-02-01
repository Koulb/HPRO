from HPRO.lcaodiag import LCAODiagKernel

kernel = LCAODiagKernel()
# FCC Diamond path: Γ → X → W → L → Γ
kernel.setk([[0.000,  0.000,  0.000],
             [0.500,  0.000,  0.500],
             [0.500,  0.250,  0.750],
             [0.500,  0.500,  0.500],
             [0.000,  0.000,  0.000]],
             [20, 15, 15, 20, 1],
             ['\u0393', 'X', 'W', 'L', '\u0393'])
kernel.load_deeph_mats('./')
kernel.diag(nbnd=20, efermi=None)
kernel.write('./')
