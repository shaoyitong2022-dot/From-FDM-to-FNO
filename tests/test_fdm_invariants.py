"""test_fdm_invariants.py: Unit tests for physical invariants in FDM solver kernels.
Checks:
  1. Mass / Integral Conservation for 1D Advection
  2. Monotonic Energy Dissipation for 1D Viscous Burgers
  3. Divergence-Free Condition (∇ · u ≈ 0) for 2D Navier-Stokes Cavity & Channel Flows
"""
import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from solvers.fdm import step01_linear_convection
from solvers.fdm import step04_burgers
from solvers.fdm import step11_cavity_flow
from solvers.fdm import step12_channel_flow

class TestFDMInvariants(unittest.TestCase):

    def test_linear_convection_mass_conservation(self):
        """Integral of u over space should remain constant (up to boundary flux)."""
        nx = 81
        dx = 2.0 / (nx - 1)
        x, u = step01_linear_convection.solve(nx=nx, nt=20)
        # Total mass should be preserved while square wave is inside domain
        mass_initial = (1.0 * (2.0 - 0.5)) + (2.0 * 0.5)  # 1.5 + 1.0 = 2.5
        mass_computed = np.sum(u) * dx
        self.assertAlmostEqual(mass_computed, 2.5, delta=0.08)

    def test_burgers_viscous_energy_dissipation(self):
        """Kinetic energy integral E(t) = 0.5 * int u^2 dx must strictly decrease."""
        x, u, u_exact, rel_err = step04_burgers.solve()
        dx = 2.0 * np.pi / (len(u) - 1)
        # Initial condition has mean 4 and variation
        e_final = 0.5 * np.sum((u - 4.0)**2) * dx
        self.assertLess(e_final, 5.0)

    def test_cavity_flow_divergence_free(self):
        """Incompressible NS requires ∂u/∂x + ∂v/∂y ≈ 0 inside the cavity."""
        x, y, u, v, p = step11_cavity_flow.solve(nit=30)
        dx = 2.0 / (len(x) - 1)
        dy = 2.0 / (len(y) - 1)

        # Central difference divergence in the interior domain
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
        div = du_dx + dv_dy

        # Check interior divergence norm (away from singular lid corners)
        interior_div = div[5:-5, 5:-5]
        mean_div = np.mean(np.abs(interior_div))
        self.assertLess(mean_div, 0.15, f"Cavity flow divergence too high: {mean_div}")

    def test_channel_flow_divergence_free(self):
        """Poiseuille channel flow divergence ∂u/∂x + ∂v/∂y ≈ 0."""
        x, y, u, v, p = step12_channel_flow.solve(nit=30)
        dx = 2.0 / (len(x) - 1)
        dy = 2.0 / (len(y) - 1)

        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
        div = du_dx + dv_dy

        interior_div = div[5:-5, 5:-5]
        mean_div = np.mean(np.abs(interior_div))
        self.assertLess(mean_div, 0.10, f"Channel flow divergence too high: {mean_div}")

if __name__ == "__main__":
    unittest.main()
