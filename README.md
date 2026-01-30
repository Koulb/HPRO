# HPRO package

HPRO is a python module for Hamiltonian Projection and Reconstruction to atomic Orbitals.

Author: Xiaoxun Gong (xiaoxun.gong@berkeley.edu)

## Installation guide

1. You should have Python 3.8+ for installing this package.
2. Go to `src/` directory and use the command `pip install .` to install the package.

## Band-RI Convergence Analysis

HPRO supports band-space Resolution of Identity (band-RI) convergence analysis for
the localized Hamiltonian. This tests how well the Hamiltonian is represented using
a truncated set of Bloch eigenstates.

### Theory

For each k-point, the localized Hamiltonian is computed as:

```
H_loc(k) = A_k† @ diag(ε_k) @ A_k
```

where:
- `A_k[n, μ] = ⟨ψ_nk|φ_μ⟩` — projection of Bloch state n onto atomic orbital μ
- `ε_k[n]` — Kohn-Sham eigenvalues

Convergence is tested by increasing the number of bands included in the sum.

### Usage

After calling `diag()`, use the `band_ri_convergence()` method:

```python
from HPRO import LCAODiagKernel

# Setup and diagonalize
kernel = LCAODiagKernel(...)
kernel.setk(...)
kernel.load_deeph_mats(...)
kernel.diag(nbnd=100)

# Run band-RI convergence analysis
result = kernel.band_ri_convergence()

# Or with custom band windows
result = kernel.band_ri_convergence(nband_list=[25, 50, 75, 100])

# Or for specific k-points only
result = kernel.band_ri_convergence(kpt_idx=[0, 5, 10])
```

### Output

The method prints convergence information and returns a dictionary:
- `nband_list`: list of band counts tested
- `errors`: array of shape (nkpts, len(nband_list)) with Frobenius norm changes
- `H_loc_final`: dict mapping k-point index to the final H_loc matrix

### Convergence Metric

The relative Frobenius norm change between successive band windows:

```
err = ||H_loc(N_i) - H_loc(N_{i-1})||_F / ||H_loc(N_i)||_F
```

