# Changelog

All notable changes to the **From FDM to FNO** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-06

### Added
- **20/20 Solver Kernel Portfolio & Automated Benchmark Suite**:
  - Track A: 12-Step Barba FDM CFD suite (Linear, Non-linear convection, Diffusion, Burgers, Laplace, Poisson, Cavity Flow, Channel Flow).
  - Track B: Foundational PyTorch Physics-Informed Neural Networks (Harmonic oscillator, Cole-Hopf shock, Poisson 2D).
  - Track C: Industrial DeepXDE implementations (1D Poisson, Harmonic oscillator with sensors, Spatio-temporal Burgers, 2D Poisson comparison).
  - Track D: Fourier Neural Operator (FNO) kernels (1D SpectralConv from scratch via rFFT, NeuralOperator 2.0+ FNO diffusion operator, sign-aware upwind FDM dataset generator).
- **Automated Verification Harness (`benchmarks/`)**:
  - `run_all_benchmarks.py`: One-click master regression harness outputting formatted academic leaderboard.
  - `analytical_solutions.py`: Ground-truth Cole-Hopf, Fourier series, and Ghia 1982 cavity data.
  - `convergence_tests.py`: Verified $p=1.00$ and $p=2.01$ empirical convergence orders.
  - `cfl_stability_scan.py`: Destructive stability scanner validating CFL Courant condition ($\sigma \le 1.0$) and von Neumann diffusion limit ($r \le 0.5$).
- **Unit Test Suite (`tests/`)**:
  - `test_fdm_invariants.py`: Mass conservation, energy dissipation, and divergence-free field checks.
  - `test_pinn_gradients.py`: Autograd computation graph validation against exact analytical derivatives.
- **5-Volume Canonical Courseware (`courseware/`)**:
  - Volume I (10 chapters): Finite difference foundations and CFD.
  - Volume II (5 chapters): Implicit fields and PINN paradigm.
  - Volume III (5 chapters): DeepXDE industrial framework practice.
  - Volume IV (5 chapters): Operator learning and Fourier Neural Operators.
  - Volume V (3 chapters): MIT 18.086 Gilbert Strang computational mathematics foundation.
  - Reference toolkits: Python tools, fluid mechanics physical intuition, bilingual scientific glossary, and Stage 0–5 learning map.
  - Tablet reader: 34 single-file offline lessons completely independent of external CDNs.
- **Project Infrastructure**:
  - Top-level `index.html` GitHub Pages interactive navigation hub.
  - GitHub Actions CI workflow for automated testing and link integrity checks.
  - Dual MIT / CC-BY-4.0 open-source licensing.
