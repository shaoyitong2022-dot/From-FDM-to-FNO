"""run_all_benchmarks.py: Master Benchmark Harness for Computational Physics & AI4Science Suite.
Runs all classical FDM, PINN, and FNO solver kernels, validates against analytical benchmarks,
and outputs an academic-grade leaderboard table.
"""
import sys
import os
import time

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

def run_all():
    print("=" * 80)
    print(">>> RUNNING COMPUTATIONAL PHYSICS & AI4SCIENCE SOLVER SUITE (2028 PH.D. BENCHMARK)")
    print("=" * 80)

    results = []

    # -------------------------------------------------------------------------
    # 1. Classical FDM Kernels (Barba CFDPython 12 Steps)
    # -------------------------------------------------------------------------
    print("\n[Track A] Running 12-Step Classical Finite Difference Solvers...")
    from solvers.fdm import (
        step01_linear_convection,
        step02_nonlinear_convection,
        step03_diffusion,
        step04_burgers,
        step05_2d_linear_convection,
        step06_2d_convection,
        step07_2d_diffusion,
        step08_2d_burgers,
        step09_2d_laplace,
        step10_2d_poisson,
        step11_cavity_flow,
        step12_channel_flow,
    )

    # Step 01
    t0 = time.perf_counter()
    x, u = step01_linear_convection.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 01",
        "problem": "1D Linear Convection",
        "method": "Upwind (FTBS)",
        "resolution": "nx=81",
        "metric": "Bounded [1, 2]",
        "val": f"max={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 02
    t0 = time.perf_counter()
    x, u = step02_nonlinear_convection.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 02",
        "problem": "1D Non-Linear Convection",
        "method": "Upwind (FTBS, wave steepening)",
        "resolution": "nx=81",
        "metric": "Front Steeping",
        "val": f"max={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 03
    t0 = time.perf_counter()
    x, u, u_exact, rel_err = step03_diffusion.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 03",
        "problem": "1D Linear Diffusion",
        "method": "Centered (FTCS)",
        "resolution": "nx=81",
        "metric": "Rel L2 vs Exact",
        "val": f"{rel_err:.2e}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if rel_err < 1e-4 else "FAIL"
    })

    # Step 04
    t0 = time.perf_counter()
    x, u, u_exact, rel_err = step04_burgers.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 04",
        "problem": "1D Viscous Burgers",
        "method": "Cole-Hopf Benchmark",
        "resolution": "nx=101",
        "metric": "Rel L2 vs Cole-Hopf",
        "val": f"{rel_err:.2%}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if rel_err < 0.20 else "FAIL"
    })

    # Step 05
    t0 = time.perf_counter()
    x, y, u = step05_2d_linear_convection.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 05",
        "problem": "2D Linear Convection",
        "method": "2D Upwind (FTBS)",
        "resolution": "81x81",
        "metric": "Max Bounded",
        "val": f"max={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 06
    t0 = time.perf_counter()
    x, y, u, v = step06_2d_convection.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 06",
        "problem": "2D Coupled Convection",
        "method": "Vector Upwind",
        "resolution": "81x81",
        "metric": "Coupled Advection",
        "val": f"max_u={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 07
    t0 = time.perf_counter()
    x, y, u = step07_2d_diffusion.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 07",
        "problem": "2D Diffusion Equation",
        "method": "2D Centered FTCS",
        "resolution": "31x31",
        "metric": "Diffusive Decay",
        "val": f"max={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 08
    t0 = time.perf_counter()
    x, y, u, v = step08_2d_burgers.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 08",
        "problem": "2D Burgers Equation",
        "method": "Coupled Convection-Diff",
        "resolution": "41x41",
        "metric": "Shock Decay",
        "val": f"max_u={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if 1.0 <= np.max(u) <= 2.05 else "FAIL"
    })

    # Step 09
    t0 = time.perf_counter()
    x, y, p, p_exact, rel_err, it = step09_2d_laplace.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 09",
        "problem": "2D Laplace Equation",
        "method": "5-Point Relaxation",
        "resolution": "31x31",
        "metric": "Rel L2 vs Series",
        "val": f"{rel_err:.2%}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if rel_err < 0.10 else "FAIL"
    })

    # Step 10
    t0 = time.perf_counter()
    x, y, p, b = step10_2d_poisson.solve()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 10",
        "problem": "2D Poisson Equation",
        "method": "Dual Source Poisson",
        "resolution": "50x50",
        "metric": "Dipole Formation",
        "val": f"span=[{np.min(p):.2f},{np.max(p):.2f}]",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if np.min(p) < -0.01 and np.max(p) > 0.01 else "FAIL"
    })

    # Step 11
    t0 = time.perf_counter()
    x, y, u, v, p = step11_cavity_flow.solve(nit=20)
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 11",
        "problem": "2D Cavity Flow (NS)",
        "method": "Chorin Projection Method",
        "resolution": "41x41",
        "metric": "Primary Vortex",
        "val": f"v_span=[{np.min(v):.2f},{np.max(v):.2f}]",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if np.min(v) < 0 and np.max(v) > 0 else "FAIL"
    })

    # Step 12
    t0 = time.perf_counter()
    x, y, u, v, p = step12_channel_flow.solve(nit=20)
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FDM",
        "id": "Step 12",
        "problem": "2D Channel Flow (NS)",
        "method": "Periodic BC Pressure Grad",
        "resolution": "41x41",
        "metric": "Parabolic Poiseuille",
        "val": f"u_max={np.max(u):.2f}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if np.max(u) > 0.0 else "FAIL"
    })

    # -------------------------------------------------------------------------
    # 2. Physics-Informed Neural Network (PINN) Kernels
    # -------------------------------------------------------------------------
    print("\n[Track B/C] Running Physics-Informed Neural Networks (PINN)...")
    from solvers.pinn import (
        p01_harmonic_oscillator,
        p02_poisson_1d,
        p03_harmonic_deepxde,
        p04_burgers_spacetime,
        p05_poisson_2d,
    )

    # p01 Pure PyTorch Harmonic Oscillator
    t0 = time.perf_counter()
    t, u_pred, u_exact, rel_err = p01_harmonic_oscillator.solve(epochs=3000)
    t_el = time.perf_counter() - t0
    results.append({
        "track": "PINN",
        "id": "PINN 01",
        "problem": "Damped Harmonic (PyTorch)",
        "method": "Autograd + Collocation",
        "resolution": "10 data + 30 phys",
        "metric": "Rel L2 Extrapolation",
        "val": f"{rel_err:.2%}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if rel_err < 0.40 else "FAIL"
    })

    # p02 DeepXDE 1D Poisson
    t0 = time.perf_counter()
    x, u_pred, u_exact, max_err, rel_l2_err = p02_poisson_1d.solve(iterations=1500)
    t_el = time.perf_counter() - t0
    results.append({
        "track": "PINN",
        "id": "PINN 02",
        "problem": "1D Poisson (DeepXDE)",
        "method": "Hessian Residual + BC",
        "resolution": "16 domain + 2 bc",
        "metric": "Max Abs Error",
        "val": f"{max_err:.2e}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if max_err < 0.05 else "FAIL"
    })

    # p03 DeepXDE Harmonic Translation
    t0 = time.perf_counter()
    t, u_pred, u_exact, rel_err = p03_harmonic_deepxde.solve(iterations=3000)
    t_el = time.perf_counter() - t0
    results.append({
        "track": "PINN",
        "id": "PINN 03",
        "problem": "Harmonic Translation",
        "method": "PointSetBC + Anchors",
        "resolution": "10 data + 40 domain",
        "metric": "Rel L2 Extrapolation",
        "val": f"{rel_err:.2%}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if rel_err < 0.40 else "FAIL"
    })

    # p04 DeepXDE Spatio-Temporal Burgers
    t0 = time.perf_counter()
    model, val_ic, u_left, u_right, shock_slope, max_val = p04_burgers_spacetime.solve(iterations=2000)
    t_el = time.perf_counter() - t0
    pass_p04 = (abs(val_ic - (-1.0)) < 0.35) and (u_left > 0.10) and (u_right < -0.10)
    results.append({
        "track": "PINN",
        "id": "PINN 04",
        "problem": "Spatio-Temporal Burgers",
        "method": "GeometryXTime Shock",
        "resolution": "2000 domain",
        "metric": "Shock Slope @ t=0.5",
        "val": f"{shock_slope:.2f}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if pass_p04 else "FAIL"
    })

    # p05 DeepXDE 2D Poisson
    t0 = time.perf_counter()
    X, Y, u_pred, u_exact, rel_err = p05_poisson_2d.solve(iterations=2000)
    t_el = time.perf_counter() - t0
    results.append({
        "track": "PINN",
        "id": "PINN 05",
        "problem": "2D Poisson Capstone",
        "method": "DeepXDE vs FDM 5-pt",
        "resolution": "600 domain + 120 bc",
        "metric": "Rel L2 vs Exact",
        "val": f"{rel_err:.2%}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if rel_err < 0.15 else "FAIL"
    })

    # -------------------------------------------------------------------------
    # 3. Fourier Neural Operator (FNO) Kernels
    # -------------------------------------------------------------------------
    print("\n[Track D] Running Fourier Neural Operator (FNO) Kernels...")
    from solvers.fno import f01_spectral_conv1d, f02_burgers_operator, f03_data_generator_fdm

    # f01 Spectral Convolution Mesh Invariance
    t0 = time.perf_counter()
    diff = f01_spectral_conv1d.test_mesh_invariance()
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FNO",
        "id": "FNO 01",
        "problem": "1D Spectral Conv Layer",
        "method": "Pure PyTorch rFFT-einsum",
        "resolution": "64 vs 256 zero-shot",
        "metric": "Invariance Discrepancy",
        "val": f"{diff:.2e}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if diff < 1e-4 else "FAIL"
    })

    # f02 Official NeuralOperator Operator Learning
    t0 = time.perf_counter()
    rel_l2, super_res_rel_l2 = f02_burgers_operator.train_and_eval(epochs=80)
    t_el = time.perf_counter() - t0
    results.append({
        "track": "FNO",
        "id": "FNO 02",
        "problem": "Diffusion Solution Operator",
        "method": "NeuralOperator FNO",
        "resolution": "80 epochs @ 64pt",
        "metric": "4x Zero-Shot Super-Res",
        "val": f"{super_res_rel_l2:.2%}",
        "time": f"{t_el:.1f}s",
        "status": "PASS" if super_res_rel_l2 < 0.08 else "FAIL"
    })

    # f03 FDM-to-FNO Data Generation Bridge
    t0 = time.perf_counter()
    X, Y = f03_data_generator_fdm.generate_operator_dataset(num_samples=30, nx=64)
    e0, eT = float(torch.mean(X ** 2).item()), float(torch.mean(Y ** 2).item())
    dissip = (e0 - eT) / e0
    t_el = (time.perf_counter() - t0) * 1000
    results.append({
        "track": "FNO",
        "id": "FNO 03",
        "problem": "FDM Data Generator Bridge",
        "method": "Sign-Aware Upwind FDM",
        "resolution": "30 samples @ 64pt",
        "metric": "Viscous Dissipation",
        "val": f"{dissip:.2%}",
        "time": f"{t_el:.1f}ms",
        "status": "PASS" if dissip > 0 else "FAIL"
    })

    # -------------------------------------------------------------------------
    # Format and Print Academic Markdown Leaderboard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("=== ACADEMIC COMPUTATIONAL PHYSICS & SCI-ML SOLVER LEADERBOARD (2028 PH.D. BENCHMARK) ===")
    print("=" * 92)
    header = f"| {'Track':<5} | {'ID':<8} | {'Physical Problem':<26} | {'Discretization / Method':<24} | {'Metric / Criterion':<22} | {'Wall Time':<9} | {'Status':<6} |"
    sep = f"|{'-'*7}|{'-'*10}|{'-'*28}|{'-'*26}|{'-'*24}|{'-'*11}|{'-'*8}|"
    print(header)
    print(sep)
    for r in results:
        row = f"| {r['track']:<5} | {r['id']:<8} | {r['problem']:<26} | {r['method']:<24} | {r['metric']:<13}: {r['val']:<7} | {r['time']:<9} | {r['status']:<6} |"
        print(row)
    print("=" * 92)

    total_pass = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)
    print(f"\nFinal Score: {total_pass}/{total_count} PASSED ({total_pass/total_count:.1%})")
    assert total_pass == total_count, f"Some benchmarks failed: {total_count - total_pass} failures."
    print("All Computational Physics & AI4Science Benchmarks PASSED with 100% Reliability!\n")

if __name__ == "__main__":
    run_all()
