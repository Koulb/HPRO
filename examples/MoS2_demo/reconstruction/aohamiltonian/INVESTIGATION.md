# MoS2 Reconstruction Investigation

## System Overview
- 2D monolayer MoS2
- 3 atoms in unit cell (1 Mo, 2 S)
- AO basis from Siesta

## Key Results

### Eigenvalue Comparison (nscf k-grid, 36 k-points)
| Bands | Original Reconstruction | Band-RI |
|-------|------------------------|---------|
| 4 | 53 meV | 18 meV |
| 8 | 44 meV | 16 meV |
| 16 | 41 meV | 39 meV |

**Key observation**: For MoS2, both methods work well! Original reconstruction has ~50 meV error, Band-RI has ~15-40 meV error.

### Band Structure Comparison (48 k-points on Γ-M-K-Γ path)
| Bands | Original | Band-RI |
|-------|----------|---------|
| 4 | 69 meV | 91 meV |
| 8 | 91 meV | 75 meV |
| 16 | 408 meV | 107 meV |

### H(R) Convergence with Number of Bands
| nbands | Band-RI vs Original error |
|--------|---------------------------|
| 20 | 62.8% |
| 50 | 38.2% |
| 100 | 9.5% |

### H(R) at Different Lattice Vectors
| R vector | ||H_orig(R)||_F | Error |
|----------|----------------|-------|
| (0,0,0) | 6.15 | 9.5% |
| (0,±1,0) | 2.53 | 5.5% |
| (±1,0,0) | 2.53 | 5.5% |

## Comparison with Diamond

| Metric | MoS2 (Original) | Diamond (Original) |
|--------|-----------------|-------------------|
| MAE (4 bands) | 53 meV | **1145 meV** |
| MAE (8 bands) | 44 meV | **1183 meV** |
| V_nl error | Small | ~1 eV systematic |

**Key finding**: MoS2 reconstruction works well with both methods, while Diamond has a systematic ~1 eV error in the Original reconstruction due to V_nl computation issues.

## Possible Explanations for MoS2 Success

1. **2D vs 3D**: MoS2 is 2D, Diamond is 3D - different periodicity
2. **Pseudopotential type**: Different elements (Mo, S vs C) have different PP characteristics
3. **Basis set overlap**: MoS2 AO basis might have better overlap with QE wavefunctions
4. **Grid integration**: 2D systems might have more accurate V_loc integration

## Generated Outputs

### Plots
1. **comparison.png** - Three-panel comparison:
   - H(R=0) convergence with number of bands
   - H(R) comparison at different R vectors
   - Eigenvalue MAE comparison (4, 8, 16 bands)

2. **band_comparison.png / band_comparison.svg** - Band structure along Γ-M-K-Γ path:
   - Red solid: QE reference
   - Blue dashed: Original reconstruction
   - Green dots: Band-RI reconstruction

## Bug Fix (2026-02-02): K-path and Fermi Energy Issues

### Problem
`plotband_comparison.py` showed inconsistent results compared to the original `plotband.py`. Visual inspection showed misaligned band structures.

### Root Causes Identified

1. **K-point conversion error**: Crystal coordinate conversion from Cartesian was incorrect
2. **Fermi energy alignment**: Using QE Fermi energy directly (4.706 eV) for all methods

### Final Solution
Rewrote `plotband_comparison.py` to:
- Use QE k-points directly from XML (Γ-K-M-Γ path at indices 0, 20, 30, 47)
- Compute Original reconstruction at those k-points
- Compute Band-RI using QE wavefunctions at matching k-points
- Use QE Fermi energy (4.706 eV) consistently for all methods

### Final Results (Band Structure Comparison)
| Bands | Original | Band-RI |
|-------|----------|---------|
| 4 | 68.8 meV | 90.6 meV |
| 8 | 91.2 meV | 75.0 meV |
| 16 | 408.2 meV | 107.1 meV |

### Key Findings
1. **For low-lying bands (4)**: Original reconstruction is slightly better (~69 meV vs ~91 meV)
2. **For mid-range bands (8)**: Band-RI becomes competitive (~75 meV vs ~91 meV)
3. **For higher bands (16)**: Band-RI is significantly better (~107 meV vs ~408 meV)

The Original reconstruction accumulates errors at higher bands due to V_nl approximations, while Band-RI maintains consistent accuracy across all bands it was constructed from.

## Conclusion

For MoS2, the real-space reconstruction works well (~50 meV error), comparable to Band-RI. This is in contrast to Diamond where the real-space reconstruction has a systematic ~1 eV error due to V_nl computation issues.

---
*Investigation complete*
