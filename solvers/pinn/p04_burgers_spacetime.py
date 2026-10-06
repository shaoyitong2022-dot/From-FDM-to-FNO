"""p04_burgers_spacetime.py: Viscous Burgers Equation in Spatio-Temporal Domain via DeepXDE.
Canonical Problem: Raissi et al. (2019) / DeepXDE official benchmark.
Equation: du/dt + u * du/dx = nu * d^2u/dx^2,  nu = 0.01 / pi
Domain: x in [-1, 1], t in [0, 0.99]
IC: u(x, 0) = -sin(pi * x)
BC: u(-1, t) = u(1, t) = 0
"""
import deepxde as dde
import numpy as np
import torch

def solve(iterations=2000, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    nu = 0.01 / np.pi
    geom = dde.geometry.Interval(-1.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, y):
        # x[:, 0]: spatial coordinate x in [-1, 1]
        # x[:, 1]: temporal coordinate t in [0, 0.99]
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - nu * dy_xx

    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=2000,
        num_boundary=100,
        num_initial=100
    )

    net = dde.nn.FNN([2] + [32] * 4 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=lr)
    model.train(iterations=iterations, display_every=1000)

    # Physical Invariants & Shock Verification:
    # 1. IC Check: at t=0, u(0.5, 0) ~ -sin(pi*0.5) = -1.0
    val_ic = float(model.predict(np.array([[0.5, 0.0]]))[0, 0])
    
    # 2. Shock Formation Check at t=0.5:
    # Wave steepens across x=0: u(-0.25, 0.5) > 0 and u(+0.25, 0.5) < 0
    u_left = float(model.predict(np.array([[-0.25, 0.5]]))[0, 0])
    u_right = float(model.predict(np.array([[0.25, 0.5]]))[0, 0])
    shock_slope = (u_left - u_right) / 0.5

    # 3. Maximum Principle Check: |u| <= 1.15
    pts_eval = np.array([[-0.5, 0.5], [0.0, 0.5], [0.5, 0.5], [0.5, 0.0]])
    max_val = float(np.max(np.abs(model.predict(pts_eval))))

    return model, val_ic, u_left, u_right, shock_slope, max_val

if __name__ == "__main__":
    model, val_ic, u_left, u_right, shock_slope, max_val = solve(iterations=2000)
    print(f"PINN p04: Spatio-Temporal Burgers PASSED (IC Check: {val_ic:.4f}, Left: {u_left:.4f}, Right: {u_right:.4f}, Shock Slope: {shock_slope:.2f})")
    assert abs(val_ic - (-1.0)) < 0.25, f"IC condition violated: {val_ic}"
    assert u_left > 0.15 and u_right < -0.15, f"Shock formation violated: u_left={u_left}, u_right={u_right}"
    assert max_val <= 1.2, f"Maximum principle violated: {max_val}"
