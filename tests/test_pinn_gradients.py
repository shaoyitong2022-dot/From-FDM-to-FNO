"""test_pinn_gradients.py: Unit tests for autograd and numerical gradient consistency in PINN models.
Verifies that higher-order automatic differentiation (Jacobian and Hessian) correctly computes
the analytical derivatives of test fields without numerical leakage.
"""
import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import deepxde as dde

class TestPINNGradients(unittest.TestCase):

    def test_autograd_1d_harmonic_derivatives(self):
        """Verify 1st and 2nd derivatives of u(t) = exp(-d*t)*cos(w*t)."""
        d, w0 = 2.0, 20.0
        w = np.sqrt(w0**2 - d**2)
        t = torch.linspace(0.1, 0.9, 50, dtype=torch.float32, requires_grad=True).view(-1, 1)

        # Function u(t)
        u = torch.exp(-d * t) * torch.cos(w * t)

        # 1st derivative via autograd
        du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        # Exact analytical 1st derivative
        du_dt_exact = -d * torch.exp(-d * t) * torch.cos(w * t) - w * torch.exp(-d * t) * torch.sin(w * t)

        diff1 = torch.max(torch.abs(du_dt - du_dt_exact)).item()
        self.assertLess(diff1, 1e-5, f"1st derivative autograd error: {diff1}")

        # 2nd derivative via autograd
        d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
        # From ODE: d2u/dt2 = -2*d*du/dt - w0^2*u
        d2u_dt2_exact = -2.0 * d * du_dt_exact - (w0**2) * u

        diff2 = torch.max(torch.abs(d2u_dt2 - d2u_dt2_exact)).item()
        self.assertLess(diff2, 1e-4, f"2nd derivative autograd error: {diff2}")

    def test_deepxde_2d_hessian_indices(self):
        """Verify deepxde.grad.hessian correctly computes diagonal vs mixed partials."""
        x = torch.tensor([[0.3, 0.4]], dtype=torch.float32, requires_grad=True)
        # Test scalar field: y = x_0^3 + 2*x_0*x_1 + 4*x_1^3
        # d^2y / dx_0^2 = 6*x_0 = 6*0.3 = 1.8
        # d^2y / dx_1^2 = 24*x_1 = 24*0.4 = 9.6
        # d^2y / dx_0 dx_1 = 2.0
        y = x[:, 0:1]**3 + 2.0 * x[:, 0:1] * x[:, 1:2] + 4.0 * x[:, 1:2]**3

        h00 = dde.grad.hessian(y, x, i=0, j=0).item()
        h11 = dde.grad.hessian(y, x, i=1, j=1).item()
        h01 = dde.grad.hessian(y, x, i=0, j=1).item()

        self.assertAlmostEqual(h00, 1.8, places=4, msg="d2y/dx0^2 incorrect")
        self.assertAlmostEqual(h11, 9.6, places=4, msg="d2y/dx1^2 incorrect")
        self.assertAlmostEqual(h01, 2.0, places=4, msg="d2y/dx0dx1 incorrect")

if __name__ == "__main__":
    unittest.main()
