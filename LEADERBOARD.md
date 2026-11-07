# Academic Computational Physics & AI4Science Solver Leaderboard

> **Automated Status**: All 20/20 Benchmarks Verified (100% PASS) | **Last Pulse**: `2026-11-07 03:50 UTC`  
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

Automated surveillance of newly published preprints covering Neural Operators, PINN improvements, and high-performance PDE solvers as of **2026-11-07 03:50 UTC**:

### 1. [LithoSAGE: Spatially Adaptive and Geometrically Equivariant Neural Operators for Computational Lithography](http://arxiv.org/abs/2611.08001v1)
- **Authors**: Ke Jin, Zhenli Xu, Zilin You (2026-11-05)
- **Abstract Takeaway**: Operator learning offers an efficient surrogate for lithography simulation, replacing costly numerical solutions with direct model inference. However, existing learning architectures are not designed ...

### 2. [Rigorously Certified Neural-Network Computation of Nonlinear Water Waves at Near-Machine Precision](http://arxiv.org/abs/2611.07395v1)
- **Authors**: Michal Shavit, Yongji Wang, Tristan Buckmaster (2026-11-05)
- **Abstract Takeaway**: We present a computational framework that combines machine-learning-based discovery with rigorous computer-assisted numerical certification and formal verification for strongly nonlinear partial diffe...

### 3. [Projection-based segregated, iterative, and multi-stage physics-informed neural network for incompressible Navier-stokes equations](http://arxiv.org/abs/2611.07387v1)
- **Authors**: Zhenyu Zhang, Zhixiang Ling, Yinghao Huo et al. (2026-11-05)
- **Abstract Takeaway**: Physics-informed neural networks(PINNs) offer a mesh-free and continuously differentiable approach to fluid simulation. However, standard monolithic PINNs for incompressible Navier-Stokes (NS) equatio...

### 4. [Fractional physics-informed neural networks to model social media virality](http://arxiv.org/abs/2611.07215v1)
- **Authors**: Alex Koong, George Em Karniadakis (2026-11-04)
- **Abstract Takeaway**: Existing models of social media virality either prescribe the functional form of engagement growth or assume a fixed mechanism by which past activity influences future engagement. We present a fractio...

### 5. [FNO-Based Inverse Co-Design of Semiconducting Quantum-Dot Devices](http://arxiv.org/abs/2611.06994v1)
- **Authors**: Marek Rychetsky, Valentin Duruisseaux, Freya Shah et al. (2026-11-04)
- **Abstract Takeaway**: Gate-defined semiconductor quantum dots are promising building blocks for scalable spin-based quantum processors, however their operational properties depend on several fabrication and control paramet...

### 6. [Unsupervised Neural Operators for Rotating Relativistic Stars: High-Fidelity Equation of State Inference and Pulsar Spin Evolution at Scale](http://arxiv.org/abs/2611.05790v1)
- **Authors**: Anarya Ray (2026-11-04)
- **Abstract Takeaway**: Observed populations of rotating neutron stars~(NSs) with millisecond-order periods are powerful probes of fundamental physics. Mass-radius measurements inform poorly constrained properties of degener...

---

## 4. How to Reproduce Locally

```bash
git clone https://github.com/shaoyitong2022-dot/From-FDM-to-FNO.git
cd From-FDM-to-FNO
python benchmarks/run_all_benchmarks.py
python scripts/daily_radar_and_leaderboard.py
```

*Generated automatically by GitHub Actions workflow `.github/workflows/daily_leaderboard.yml`.*
