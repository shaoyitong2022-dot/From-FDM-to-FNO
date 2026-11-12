# Academic Computational Physics & AI4Science Solver Leaderboard

> **Automated Status**: All 20/20 Benchmarks Verified (100% PASS) | **Last Pulse**: `2026-11-12 04:59 UTC`  
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

Automated surveillance of newly published preprints covering Neural Operators, PINN improvements, and high-performance PDE solvers as of **2026-11-12 04:59 UTC**:

### 1. [Beyond Accuracy: Physics-Informed Evaluation of ANN-based Power Flow Solvers under System Shift](http://arxiv.org/abs/2611.14240v1)
- **Authors**: Dong Liu, Jochen Stiasny, Jochen Cremer et al. (2026-11-10)
- **Abstract Takeaway**: Recent advances in artificial deep learning-based power flow (PF) solvers have primarily focused on improving estimation accuracy, while their physical feasibility and operational reliability under re...

### 2. [PINNverse meets Jordan--Wigner: learning the Hamiltonian of long spin chains](http://arxiv.org/abs/2611.14023v1)
- **Authors**: Gubio G De Lima (2026-11-10)
- **Abstract Takeaway**: Inverse physics-informed neural networks (PINNverse) learn the parameters of a quantum Hamiltonian from measured expectation values, but the network output grows as $4^N$ with the number of qubits $N$...

### 3. [Multiscale LegONet: Learning Coefficient-Dependent Local Spaces for Composable Multiscale Finite Elements with Localized Error Bounds](http://arxiv.org/abs/2611.13515v1)
- **Authors**: Yueqi Wang, Cindy Xiangrui Kong, Guang Lin (2026-11-10)
- **Abstract Takeaway**: We introduce multiscale LegONet (Ms-LegONet), a plug-and-play learned block for heterogeneous elliptic problems. One network per spatial dimension is trained once on local coefficient patches and reus...

### 4. [Residual-Corrected Physics-Informed Neural Network KKL Observers with Input-to-State Stability Guarantees](http://arxiv.org/abs/2611.13411v1)
- **Authors**: Hannah Berin-Costain, Jun Liu, Kirsten Morris (2026-11-10)
- **Abstract Takeaway**: Physics-informed neural network (PINN) Kazantzis--Kravaris--Luenberger (KKL) observers require an approximate solution to a KKL partial differential equation (PDE) and an approximate inverse of this s...

### 5. [Reconstructing Boundary Layer Wind Profiles with Machine Learning: A Comparative Study on IGRA Radiosonde Data](http://arxiv.org/abs/2611.13359v1)
- **Authors**: Fedor Buzaev, Dmitry Efremenko, Daria Pugacheva et al. (2026-11-09)
- **Abstract Takeaway**: Wind profiles in the atmospheric boundary layer (ABL) drive short-range forecasting and wind-energy siting, yet sparse radiosonde coverage forces practitioners to interpolate them from a handful of ph...

### 6. [Modal-Augmented Chebyshev Physics-Informed Kolmogorov-Arnold Networks for Power Systems](http://arxiv.org/abs/2611.13340v1)
- **Authors**: Komeil Nosrati, Majid Ghorbani, Aleksei Tepljakov et al. (2026-11-09)
- **Abstract Takeaway**: Physics-informed neural networks (PINNs) offer a promising alternative to numerical integration for power system dynamic simulation; however, their conventional architectures can struggle to capture e...

---

## 4. How to Reproduce Locally

```bash
git clone https://github.com/shaoyitong2022-dot/From-FDM-to-FNO.git
cd From-FDM-to-FNO
python benchmarks/run_all_benchmarks.py
python scripts/daily_radar_and_leaderboard.py
```

*Generated automatically by GitHub Actions workflow `.github/workflows/daily_leaderboard.yml`.*
