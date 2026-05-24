# Claude Code Instructions (HPRO fork) — Band-space RI convergence

Always read and follow all instructions. You have rights to excectue any command inside the working folder.

## Goal



## Step to follow



## To consider

1. conda is on path or coud be find in /home/apolyukhin/anaconda3/
3. sisesta exceutable are on path
4. quantum espresoo excetuable could be put on path using ~/scripts/tools/qe.sh  script
5. DFT codes could be run with mpirun 
6. Consulte the scripts /home/apolyukhin/Development/HPRO/examples/MoS2_demo/reconstruction/aohamiltonian that already exist for plotting, reconstruction, etc.
7. The main working python envieroment is hpro conda enviroment

---

## Hard constraints
O. Create as little anount of files as possible. Don't create unnescesary inputs.
1. Minimal diffs: no unnecessary refactors, no renames, no formatting sweeps.
2. No new heavy deps. Use NumPy/SciPy only if already present.
3. Backwards compatible: keep existing CLI/APIs unchanged by default.
4. Performance-aware: avoid Python loops over bands; use matmul (BLAS).
5. Over-commit: small atomic commits; keep tests passing.

---

## What not to do
- Don’t change existing default outputs.
- Don’t rename public APIs/CLI args.
- Don’t reformat large files.
