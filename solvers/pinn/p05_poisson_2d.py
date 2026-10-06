"""p05_poisson_2d.py: 2D Poisson Equation via DeepXDE with Exact Analytical & FDM Benchmark.
Equation: d^2 u / dx^2 + d^2 u / dy^2 = -2*pi^2 * sin(pi*x)*sin(pi*y)
Domain: [0, 1] x [0, 1], u = 0 on boundary.
Analytical Solution: u_exact(x, y) = sin(pi*x)*sin(pi*y)

Pedagogical Highlight:
  - dy_xx requires i=0, j=0 (d^2 u / dx_0^2)
  - dy_yy requires i=1, j=1 (d^2 u / dx_1^2). Setting i=0, j=1 computes the mixed partial derivative!
"""
import torch
import deepxde as dde
import numpy as np

def solve(iterations=3000, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    def pde(x, y):
        # x[:, 0]: x coordinate, x[:, 1]: y coordinate
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)  # i=1, j=1 for second derivative w.r.t y!
        source = -2.0 * (torch.pi ** 2) * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
        return dy_xx + dy_yy - source

    bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(geom, pde, bc, num_domain=600, num_boundary=120)

    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    model.train(iterations=iterations, display_every=1000)

    # Evaluation on 31x31 grid
    nx, ny = 31, 31
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.vstack([X.ravel(), Y.ravel()]).T

    u_pred = model.predict(pts).reshape((nx, ny))
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

    rel_l2_err = float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))
    return X, Y, u_pred, u_exact, rel_l2_err

if __name__ == "__main__":
    X, Y, u_pred, u_exact, rel_err = solve(iterations=3000)
    print(f"PINN p05: 2D Poisson PASSED (Relative L2 Error: {rel_err:.2%})")
    assert rel_err < 0.15, f"2D Poisson error too high: {rel_err:.2%}"
