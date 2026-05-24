# Diamond Reconstruction Investigation

## Problem Statement
The real-space reconstruction for Diamond shows ~1.1-1.3 eV eigenvalue error vs QE reference, while Band-RI reconstruction gives only ~60 meV error.

## Key Findings

### 1. K-point Folding is NOT the Issue
- Tested: H(k) at k=(0.75,0.75,0.75) vs k=(-0.25,-0.25,-0.25)
- Result: `||H(k_orig) - H(k_folded)||_F = 6.79e-16`
- Conclusion: H(k) is properly periodic in reciprocal space

### 2. Error is Systematic Across ALL K-points
| Metric | Original Reconstruction | Band-RI |
|--------|------------------------|---------|
| Min error | 1061.7 meV | 33.2 meV |
| Max error | 1330.4 meV | 74.2 meV |
| Mean error | 1145.3 meV | 59.5 meV |
| Std dev | 46.2 meV | 11.4 meV |

### 3. Eigenvalue Shift at Gamma
| Band | QE (eV) | Original (eV) | Band-RI (eV) | Orig Error |
|------|---------|---------------|--------------|------------|
| 1 | -8.96 | -8.00 | -8.92 | +960 meV |
| 2-4 | 11.96 | 13.41 | 11.99 | +1454 meV |
| 5-7 | 17.95 | 19.04 | 17.98 | +1095 meV |

**Key observation**: Original eigenvalues are systematically HIGHER than QE by ~1-1.5 eV.

### 4. Hamiltonian Construction
From `kernel.py`, the Hamiltonian is:
```
H = Hkin + Hmain + Hcorr
```
Where:
- `Hkin` = kinetic energy ⟨φ|−∇²/2|φ⟩
- `Hmain` = local potential ⟨φ|V_loc|φ⟩
- `Hcorr` = nonlocal PP ⟨φ|V_nl|φ⟩

## Hypotheses to Test

### Hypothesis A: Local Potential Normalization
The VSC file contains V_eff(G). The FFT normalization could be wrong.
- `read_vloc()` uses `norm='forward'` in ifftn
- Need to check if dvol factor is applied correctly

### Hypothesis B: Kinetic Energy Issue
The kinetic energy matrix might have wrong units or normalization.

### Hypothesis C: Nonlocal PP Issue
The D_ij coefficients or projector functions might be wrong.

## Test Results

### Test 1: Matrix Comparison at Gamma
```
||H_file - H_bandRI||_F = 0.282
Tr(H_file - H_bandRI) = 24.98 eV  (average per orbital: 0.96 eV)
```

The difference is NOT proportional to overlap S (residual = 65%).

### Test 2: Band-dependent Errors
| Band | QE (eV) | File Error | BandRI Error |
|------|---------|------------|--------------|
| 1 (s-like) | -8.96 | +960 meV | +41 meV |
| 2-4 (p-like) | 11.96 | +1454 meV | +31 meV |
| 5-7 | 17.95 | +1095 meV | +34 meV |
| 8 | 25.29 | +1899 meV | +268 meV |

**Key insight**: Error varies by band character, NOT a constant shift!

### Test 3: Constant Offset Correction
Subtracting 960 meV (band 1 error) reduces MAE from 1330 → 371 meV.
Still significant error for higher bands.

### Test 4: A Matrix Quality
- `||A† A - S||_F = 0.01` - excellent!
- Band space is incomplete (expected with 100 bands)

## V_loc Normalization Test Results

**Finding**: V_loc normalization is almost correct!
- Optimal factor / standard dvol = **1.035** (only 3.5% off)
- This is NOT the main source of error

**Even with optimal V_loc, significant errors remain**:
| Band | Character | Error with optimal V_loc |
|------|-----------|--------------------------|
| 1 | s-like | 502 meV |
| 2-4 | p-like | -30 meV |
| 5-7 | mixed | 47 meV |
| 8 | higher | 92 meV |

**Key insight**: Band 1 (s-like) has much larger error than p-like bands!

## New Hypothesis: Nonlocal Pseudopotential (V_nl) Issue

The D_ij coefficients from pseudopotential:
```
D_ij diagonal: [6.48, 0.39, -4.20, -4.20, -4.20, -0.88, -0.88, -0.88] Hartree
                 s1    s2      p1     p2     p3     p4     p5     p6
```

The s-channel has large positive D values (6.48 Ha ≈ 176 eV), which could cause issues if projector-AO overlaps are wrong.

## Evidence Summary

1. **V_loc normalization** is approximately correct (ratio ≈ 1.03)
2. **Error varies by orbital character** - s-like bands have larger errors
3. **V_nl contribution** is likely the source of remaining error
4. **D_ij values** are large for s-channels

## ROOT CAUSE IDENTIFIED: V_nl is Systematically Too High

### Key Finding
The V_nl computed in real-space reconstruction is systematically **higher** than what Band-RI implies:

| Orbital | V_nl_file (eV) | V_nl_bandRI (eV) | Difference |
|---------|---------------|-----------------|------------|
| 1 (s) | 7.28 | 5.70 | +1.58 eV |
| 2 (s) | 14.81 | 11.88 | +2.92 eV |
| 3-5 (p) | -2.21 | -2.89 | +0.67 eV |
| 6-8 (p) | -7.49 | -8.61 | +1.12 eV |

### Validation
Using V_nl_bandRI instead of V_nl_file:
- **Before**: MAE = 1330 meV
- **After**: MAE = 33 meV (same as Band-RI!)

### Trace Analysis
```
Tr(V_nl_file) - Tr(V_nl_bandRI) = 24.98 eV
Per orbital: 0.96 eV  <-- matches the systematic ~1 eV error!
```

## Conclusion
The V_nl contribution from real-space reconstruction is too positive by ~1 eV.
This shifts all eigenvalues UP by approximately 1 eV.

## Possible Causes
1. **Projector normalization** - β(r) functions might be scaled incorrectly
2. **D_ij units** - might have wrong factor (Ry vs Ha already checked)
3. **Overlap integral** - ⟨φ|β⟩ computation might have FFT/grid issues
4. **Missing factor** - some normalization constant might be missing

## For Diamond Example
The real-space reconstruction currently has this systematic V_nl error.
Band-RI reconstruction avoids this issue by using QE eigenvalues directly.

## Generated Outputs

### Plots
1. **comparison.png** - Three-panel comparison:
   - H(R=0) convergence with number of bands
   - H(R) comparison at different R vectors
   - Eigenvalue MAE comparison (4, 8, 16 bands)

2. **band.png / band.svg** - Band structure along Γ-X-W-L-Γ path:
   - Red solid: QE reference
   - Blue dashed: Original reconstruction
   - Green dots: Band-RI reconstruction

### Numerical Results

**Eigenvalue MAE vs QE (nscf k-grid, 64 k-points)**:
| Bands | Original | Band-RI |
|-------|----------|---------|
| 4 | 1145 meV | 60 meV |
| 8 | 1183 meV | 66 meV |
| 16 | 4068 meV | 1690 meV |

**Band Structure MAE vs QE (71 k-points on path)**:
| Bands | Original | Band-RI |
|-------|----------|---------|
| 4 | 1110 meV | 52 meV |
| 8 | 1202 meV | 68 meV |
| 16 | 4283 meV | 1978 meV |

---
*Investigation complete: Root cause is V_nl computation error in real-space reconstruction*
