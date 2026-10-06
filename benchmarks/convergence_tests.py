"""convergence_tests.py: Automated Mesh Refinement and Convergence Order Verification.
Computes empirical convergence order:
    p = ln(E(h1) / E(h2)) / ln(h1 / h2)
Verifies:
  - 1D Convection (Step 01 Upwind): O(Δx) -> p ≈ 1.0
  - 1D Diffusion (Step 03 FTCS): O(Δx^2) -> p ≈ 2.0
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from solvers.fdm import step03_diffusion

def test_convection_convergence():
    """Verify O(Δx) spatial accuracy of 1st-order Upwind on smooth sine wave."""
    c = 1.0
    L = 2.0
    T = 0.5
    sigma = 0.5
    grids = [41, 81, 161, 321]
    errors = []
    dxs = []

    for nx in grids:
        dx = L / (nx - 1)
        dt = sigma * dx / c
        nt = int(round(T / dt))
        actual_T = nt * dt
        x = np.linspace(0, L, nx)
        u = np.sin(2.0 * np.pi * x / L)
        
        for _ in range(nt):
            un = u.copy()
            u[1:] = un[1:] - sigma * (un[1:] - un[:-1])
            u[0] = un[0] - sigma * (un[0] - un[-2])
            
        u_exact = np.sin(2.0 * np.pi * (x - c * actual_T) / L)
        l2_err = np.sqrt(dx * np.sum((u - u_exact) ** 2))
        errors.append(l2_err)
        dxs.append(dx)

    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(dxs[i] / dxs[i + 1])
        for i in range(len(grids) - 1)
    ]
    return dxs, errors, orders

def test_diffusion_convergence():
    """Verify O(Δx^2) spatial accuracy of FTCS on 1D diffusion equation."""
    dt = 1e-5
    T = 0.05
    grids = [41, 81, 161]
    errors = []
    dxs = []

    for nx in grids:
        dx = 2.0 / (nx - 1)
        dxs.append(dx)
        x, u, u_exact, rel_err = step03_diffusion.solve(nx=nx, T=T, dt=dt)
        l2_err = np.sqrt(dx * np.sum((u - u_exact) ** 2))
        errors.append(l2_err)

    orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(dxs[i] / dxs[i + 1])
        for i in range(len(grids) - 1)
    ]
    return dxs, errors, orders

if __name__ == "__main__":
    print("=== Convergence Test 1: 1D Convection (Upwind / FTBS) ===")
    dxs_c, errs_c, orders_c = test_convection_convergence()
    for dx, e in zip(dxs_c, errs_c):
        print(f"  dx = {dx:.4f} -> L2 error = {e:.4e}")
    for i, p in enumerate(orders_c):
        print(f"  Refinement {i+1}->{i+2}: Observed p = {p:.2f} (Theoretical: 1.0)")
    assert 0.95 <= orders_c[-1] <= 1.05, f"Convection order p={orders_c[-1]:.2f} deviates from 1.0"

    print("\n=== Convergence Test 2: 1D Diffusion (Centered FTCS) ===")
    dxs_d, errs_d, orders_d = test_diffusion_convergence()
    for dx, e in zip(dxs_d, errs_d):
        print(f"  dx = {dx:.4f} -> L2 error = {e:.4e}")
    for i, p in enumerate(orders_d):
        print(f"  Refinement {i+1}->{i+2}: Observed p = {p:.2f} (Theoretical: 2.0)")
    assert 1.90 <= orders_d[-1] <= 2.10, f"Diffusion order p={orders_d[-1]:.2f} deviates from 2.0"

    print("\nAll Convergence Tests PASSED (First-order & Second-order verified).")
