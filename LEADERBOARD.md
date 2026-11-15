# Academic Computational Physics & AI4Science Solver Leaderboard

> **Automated Status**: All 20/20 Benchmarks Verified (100% PASS) | **Last Pulse**: `2026-11-15 03:56 UTC`  
> **Project Engine**: [From-FDM-to-FNO](https://github.com/shaoyitong2022-dot/From-FDM-to-FNO) | **Maintainer**: Yitong (SYSU Physics)

This leaderboard provides a transparent, empirical performance comparison across **Classical Finite Difference (FDM)**, **Physics-Informed Neural Networks (PINN)**, and **Fourier Neural Operators (FNO)** under identical physical initial-boundary value conditions.

---

## 1. Cross-Method Architectural Comparison Matrix

| Evaluation Dimension | Classical FDM (Barba/LeVeque) | Physics-Informed NN (PINN) | Fourier Neural Operator (FNO) |
|---|---|---|---|
| **Problem Scope** | Solves **1 instance** of boundary problem | Solves **1 instance** (forward or inverse) | Learns mapping over **function space** |
| **Evaluation Speed** | Milliseconds ($0.1\text{ms} \sim 215\text{ms}$) | Seconds ($8.5\text{s} \sim 19.8\text{s}$) | Milliseconds inference ($1.6\text{s}$ train, $<10\text{ms}$ eval) |
| **Exact Conservation** | **Strict** (Discrete flux conservation) | Approximate (Soft penalty in loss) | Approximate (Learned continuous kernel) |
| **Mesh Dependence** | Bound to discrete spatial grid $\Delta x$ | Continuous mesh-free collocation | **Mesh-Independent** (Zero-shot super-resolution) |
| **Optimal Industrial Role** | High-precision forward verification solver | Sparse data sensor fusion & inverse parameter recovery | Fast surrogate simulation, real-time control, digital twins |

---

## 2. Benchmark Regression Leaderboard (20/20 Passed)

| Track | Kernel ID | Physical Problem | Discretization / Method | Verification Metric / Criterion | Wall Time | Status |
|:---:|:---:|---|---|---|:---:|:---:|
| **FDM** | `Step 01` | 1D Linear Convection | Upwind (FTBS) | Bounded [1, 2]: max=2.00 | `0.8ms` | ✅ PASS |
| **FDM** | `Step 02` | 1D Non-Linear Convection | Upwind (FTBS, wave steepening) | Front Steeping: max=2.00 | `0.1ms` | ✅ PASS |
| **FDM** | `Step 03` | 1D Linear Diffusion | Centered (FTCS) | Rel L2 vs Exact: 1.02e-05 | `1.9ms` | ✅ PASS |
| **FDM** | `Step 04` | 1D Viscous Burgers | Cole-Hopf Benchmark | Rel L2 vs Cole-Hopf: 16.81% | `214.6ms` | ✅ PASS |
| **FDM** | `Step 05` | 2D Linear Convection | 2D Upwind (FTBS) | Max Bounded: max=1.98 | `2.5ms` | ✅ PASS |
| **FDM** | `Step 06` | 2D Coupled Convection | Vector Upwind | Coupled Advection: max_u=1.99 | `9.9ms` | ✅ PASS |
| **FDM** | `Step 07` | 2D Diffusion Equation | 2D Centered FTCS | Diffusive Decay: max=1.77 | `0.4ms` | ✅ PASS |
| **FDM** | `Step 08` | 2D Burgers Equation | Coupled Convection-Diff | Shock Decay: max_u=2.00 | `8.1ms` | ✅ PASS |
| **FDM** | `Step 09` | 2D Laplace Equation | 5-Point Relaxation | Rel L2 vs Series: 7.81% | `36.8ms` | ✅ PASS |
| **FDM** | `Step 10` | 2D Poisson Equation | Dual Source Poisson | Dipole Formation: span=[-0.05,0.05] | `2.1ms` | ✅ PASS |
| **FDM** | `Step 11` | 2D Cavity Flow (NS) | Chorin Projection Method | Primary Vortex: v_span=[-0.25,0.23] | `215.7ms` | ✅ PASS |
| **FDM** | `Step 12` | 2D Channel Flow (NS) | Periodic BC Pressure Grad | Parabolic Poiseuille: u_max=0.17 | `55.6ms` | ✅ PASS |
| **PINN** | `PINN 01` | Damped Harmonic (PyTorch) | Autograd + Collocation | Rel L2 Extrapolation: 28.90% | `19.8s` | ✅ PASS |
| **PINN** | `PINN 02` | 1D Poisson (DeepXDE) | Hessian Residual + BC | Max Abs Error: 3.54e-04 | `8.5s` | ✅ PASS |
| **PINN** | `PINN 03` | Harmonic Translation | PointSetBC + Anchors | Rel L2 Extrapolation: 29.22% | `17.3s` | ✅ PASS |
| **PINN** | `PINN 04` | Spatio-Temporal Burgers | GeometryXTime Shock | Shock Slope @ t=0.5: 2.82 | `11.3s` | ✅ PASS |
| **PINN** | `PINN 05` | 2D Poisson Capstone | DeepXDE vs FDM 5-pt | Rel L2 vs Exact: 10.08% | `16.2s` | ✅ PASS |
| **FNO** | `FNO 01` | 1D Spectral Conv Layer | Pure PyTorch rFFT-einsum | Invariance Discrepancy: 1.51e-07 | `65.9ms` | ✅ PASS |
| **FNO** | `FNO 02` | Diffusion Solution Operator | NeuralOperator FNO | 4x Zero-Shot Super-Res: 1.67% | `1.6s` | ✅ PASS |
| **FNO** | `FNO 03` | FDM Data Generator Bridge | Sign-Aware Upwind FDM | Viscous Dissipation: 75.92% | `164.1ms` | ✅ PASS |

---

## 3. Daily AI4Science & SciML Research Radar (arXiv Pulse)

Automated surveillance of newly published preprints covering Neural Operators, PINN improvements, and high-performance PDE solvers as of **2026-11-15 03:56 UTC**:

### 1. [Well-Balanced WE-PINNs for the Shallow Water Equations over Smooth and Discontinuous Bathymetries](http://arxiv.org/abs/2611.17289v1)
- **Authors**: Ismail Oubarka, Imad Kissami, Mohamed Boubekeur et al. (2026-11-12)
- **Abstract Takeaway**: Physics-informed neural networks (PINNs) solve partial differential equations without a mesh, but they do not in general preserve the steady states that well-balanced schemes reproduce exactly. We pre...

### 2. [Media Trading with Censored Auction Prices: A Stochastic Control and Reinforcement Learning Approach](http://arxiv.org/abs/2611.16992v1)
- **Authors**: Yun-Ithry Gamrani, Huyên Pham (2026-11-12)
- **Abstract Takeaway**: In real-time bidding, marketing agencies must balance campaign performance, transaction margins and the use of a finite advertising budget. A key difficulty is that market-clearing prices are typicall...

### 3. [Recurrent Fourier Neural Operator for Event-by-Event Relativistic Hydrodynamics](http://arxiv.org/abs/2611.16511v1)
- **Authors**: Xiangrun Bao, Lingxiao Wang, Huichao Song (2026-11-12)
- **Abstract Takeaway**: Relativistic hydrodynamics is a powerful tool to simulate the evolution of the quark--gluon plasma (QGP) in relativistic heavy-ion collisions, but event-by-event simulations are computationally demand...

### 4. [A Lower-envelope Eikonal Neural Operator](http://arxiv.org/abs/2611.16240v1)
- **Authors**: Hwijae Son, Sung Woong Cho, Jooyoung Hahn (2026-11-12)
- **Abstract Takeaway**: We propose the Lower-envelope Eikonal Neural Operator (LENO), a physics-informed neural operator for eikonal equations with heterogeneous velocity fields and general source sets. LENO encodes the vari...

### 5. [Newton's Method and Policy Iterations for Viscous Hamilton--Jacobi Equations](http://arxiv.org/abs/2611.15678v1)
- **Authors**: Diogo A. Gomes, Xiaoqin Guo, Yeoneung Kim et al. (2026-11-11)
- **Abstract Takeaway**: We study Newton's method for the Cauchy problem for viscous Hamilton--Jacobi equations with Hamiltonians that need not be convex. Each Newton step solves a linear parabolic equation obtained by linear...

### 6. [Profile-Based Auditing of Domestic Violence Survey Selection Across South Asia](http://arxiv.org/abs/2611.15106v1)
- **Authors**: Mohammad Abu Obaida Mullick, Rezaur Rahman Ratul, Mymuna Nourin et al. (2026-11-11)
- **Abstract Takeaway**: Domestic violence (DV) data in the Demographic and Health Surveys (DHS) are collected through a specialized module whose household-level sampling and privacy requirements create a distinct observation...

---

## 4. How to Reproduce Locally

```bash
git clone https://github.com/shaoyitong2022-dot/From-FDM-to-FNO.git
cd From-FDM-to-FNO
python benchmarks/run_all_benchmarks.py
python scripts/daily_radar_and_leaderboard.py
```

*Generated automatically by GitHub Actions workflow `.github/workflows/daily_leaderboard.yml`.*
