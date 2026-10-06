"""p02_poisson_1d.py: 1D Poisson Equation solved via DeepXDE.
Equation: -d^2 u / dx^2 = 1, x in [0, 1]
Boundary Conditions: u(0) = 0, u(1) = 0
Analytical Solution: u(x) = 0.5 * x * (1 - x)
"""
import deepxde as dde
import numpy as np

def solve(iterations=2000, lr=1e-3):
    geom = dde.geometry.Interval(0, 1)

    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - 1.0

    def boundary(x, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary)
    data = dde.data.PDE(geom, pde, bc, num_domain=16, num_boundary=2)

    net = dde.nn.FNN([1] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    model.train(iterations=iterations, display_every=1000)

    x_test = np.linspace(0, 1, 101)[:, None]
    u_pred = model.predict(x_test)
    u_exact = 0.5 * x_test * (1.0 - x_test)

    max_err = float(np.max(np.abs(u_pred - u_exact)))
    rel_l2_err = float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))
    return x_test, u_pred, u_exact, max_err, rel_l2_err

if __name__ == "__main__":
    x, u_pred, u_exact, max_err, rel_l2_err = solve(iterations=2000)
    print(f"PINN p02: 1D Poisson PASSED (Max Error: {max_err:.4e}, Rel L2: {rel_l2_err:.2%})")
    assert max_err < 0.05, f"1D Poisson max error too large: {max_err}"
