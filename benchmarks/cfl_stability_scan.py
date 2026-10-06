"""cfl_stability_scan.py: Destructive Stability Boundary & Von Neumann Condition Scanner.
Validates the theoretical stability limits derived in MIT 18.086:
  1. Courant-Friedrichs-Lewy (CFL) Condition for Upwind Convection:
     σ = c Δt / Δx <= 1.0  (Stable)
     σ > 1.0               (Explosive Blowup)
  2. Von Neumann Stability Condition for FTCS Diffusion:
     r = ν Δt / Δx^2 <= 0.5 (Stable Monotonic Decay)
     r > 0.5                (High-Frequency Oscillatory Blowup)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

def scan_convection_cfl():
    """Scan Courant number σ across the critical boundary σ=1.0."""
    c = 1.0
    L = 2.0
    nx = 81
    dx = L / (nx - 1)
    sigmas = [0.5, 0.8, 1.0, 1.05, 1.2]
    results = {}

    for sigma in sigmas:
        dt = sigma * dx / c
        nt = 60
        u = np.ones(nx)
        u[int(0.5 / dx):int(1.0 / dx + 1)] = 2.0
        blown_up = False

        for _ in range(nt):
            un = u.copy()
            u[1:] = un[1:] - sigma * (un[1:] - un[:-1])
            u[0] = 1.0
            if np.isnan(u).any() or np.isinf(u).any() or np.max(np.abs(u)) > 10.0:
                blown_up = True
                break

        max_val = np.max(np.abs(u)) if not blown_up else float("inf")
        results[sigma] = {"blown_up": blown_up, "max_val": max_val}

    return results

def scan_diffusion_r():
    """Scan diffusion number r across the critical boundary r=0.5."""
    nu = 0.1
    L = 2.0
    nx = 41
    dx = L / (nx - 1)
    r_values = [0.2, 0.4, 0.5, 0.55, 0.7]
    results = {}

    for r in r_values:
        dt = r * (dx ** 2) / nu
        nt = 50
        u = np.ones(nx)
        u[int(0.5 / dx):int(1.0 / dx + 1)] = 2.0
        blown_up = False

        for _ in range(nt):
            un = u.copy()
            u[1:-1] = un[1:-1] + r * (un[2:] - 2.0 * un[1:-1] + un[:-2])
            u[0] = un[0] + r * (un[1] - 2.0 * un[0] + un[-2])
            u[-1] = u[0]
            if np.isnan(u).any() or np.isinf(u).any() or np.max(np.abs(u)) > 10.0:
                blown_up = True
                break

        max_val = np.max(np.abs(u)) if not blown_up else float("inf")
        results[r] = {"blown_up": blown_up, "max_val": max_val}

    return results

if __name__ == "__main__":
    print("=== CFL Stability Boundary Scan: 1D Convection (FTBS) ===")
    res_cfl = scan_convection_cfl()
    for s, info in res_cfl.items():
        status = "BLOWUP (Unstable)" if info["blown_up"] else f"STABLE (max |u| = {info['max_val']:.2f})"
        print(f"  σ = {s:4.2f} -> {status}")
    assert not res_cfl[1.0]["blown_up"], "σ=1.0 should be stable!"
    assert res_cfl[1.2]["blown_up"], "σ=1.2 should blow up!"

    print("\n=== Von Neumann Stability Boundary Scan: 1D Diffusion (FTCS) ===")
    res_diff = scan_diffusion_r()
    for r, info in res_diff.items():
        status = "BLOWUP (Unstable)" if info["blown_up"] else f"STABLE (max |u| = {info['max_val']:.4f})"
        print(f"  r = {r:4.2f} -> {status}")
    assert not res_diff[0.5]["blown_up"], "r=0.5 should be stable!"
    assert res_diff[0.7]["blown_up"], "r=0.7 should blow up!"

    print("\nAll Stability Boundary Scans PASSED (Strictly verifies MIT 18.086 stability criteria).")
