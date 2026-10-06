# Project Roadmap & Track Status

This document tracks the current implementation state, verified benchmarks, and future development horizons for the **From FDM to FNO** suite.

---

## 1. Master Track Status Overview

| Track | Theme & Focus | Status | Lessons | Solvers / Kernels | Benchmark Coverage |
|---|---|---|---|---|---|
| **Track A** | Classical FDM & Incompressible CFD (Barba 12 Steps) | ✅ Completed | 10 Chapters | 12 Kernels | 12/12 PASS (Ghia 1982, Cole-Hopf) |
| **Track B** | Foundational PINN (Autograd, Loss Balancing, OOD) | ✅ Completed | 5 Chapters | 5 Kernels | 5/5 PASS (Analytical comparisons) |
| **Track C** | Industrial DeepXDE Framework (Hessian, L-BFGS) | ✅ Completed | 5 Chapters | Integrated in B | 5/5 PASS (Dual-track cross-validation) |
| **Track D** | Neural Operators & FNO (Caltech spectral convolution) | ✅ Completed | 5 Chapters | 3 Kernels | 3/3 PASS (4x Zero-Shot Super-Res) |
| **Track E** | Computational Mathematics (Gilbert Strang MIT 18.086) | ✅ Completed | 3 Chapters | Theory & Matrices | 2/2 Stability & Convergence Scans |
| **Track R** | Research Methodology & Verification Diagnostics | 🔄 In Progress | 4 Chapters | Baseline Suites | Leaderboard Regression |

---

## 2. Completed Milestones (v1.0.0)

- [x] **20/20 Automated Solver Benchmark Leaderboard**:
  - Full regression harness executing in <70s total wall time.
  - Analytical verification against Cole-Hopf (Burgers) and Ghia 1982 (Cavity Flow).
  - Empirical convergence rate verification ($p=1.00$ for Upwind, $p=2.01$ for FTCS).
  - Destructive stability scan verifying CFL Courant $\sigma \le 1.0$ and von Neumann diffusion $r \le 0.5$.
- [x] **Canonical 28-Chapter Open-Source Courseware**:
  - Full volume set (Volume I to Volume V).
  - Formula-to-code line-by-line mapping with strictly localized MathJax rendering.
- [x] **Zero-CDN Standalone Offline Architecture**:
  - 34 single-file offline HTML lessons (`tablet/`) runnable without internet connection.
  - Pre-bundled localized MathJax and typography assets.

---

## 3. Near-Term Horizon (2026 Winter – 2027 Spring)

- [ ] **Track F: MicroFDTD-Plasma (Modern C++ & CUDA)**:
  - 1D & 2D Yee grid electromagnetic wave solver based on John B. Schneider's *Understanding the FDTD Method*.
  - Drude model cold plasma auxiliary differential equation (ADE) implementation.
  - Cutoff frequency reflection and transmission verification against analytical dielectric boundaries.
- [ ] **Track G: Neural-Preconditioned Krylov Solvers**:
  - Coupling lightweight Fourier Neural Operators (FNO) as learned preconditioners for classical Conjugate Gradient (CG) Poisson solvers.
  - Demonstrating measurable speedup: iteration count reduction from $>150$ to $<15$ while maintaining exact machine-precision conservation.
- [ ] **NVIDIA Warp Accelerated Kernel Port**:
  - Pure Python JIT CUDA kernels for 2D Navier-Stokes and plasma advection.

---

## 4. Community Feedback & Track Activation Rules

- Scope expansion is strictly regulated to prevent feature bloat.
- New solver tracks are triggered by documented scientific need, peer-reviewed benchmarks, or verified upstream compatibility (e.g. PyTorch / DeepXDE API upgrades).
