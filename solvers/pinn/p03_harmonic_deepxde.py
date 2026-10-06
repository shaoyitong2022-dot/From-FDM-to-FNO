"""p03_harmonic_deepxde.py: Damped Harmonic Oscillator via DeepXDE.
Demonstrates translation of external observation data via PointSetBC and anchors.
Equation: d^2 u / dt^2 + 2*d * du/dt + w0^2 * u = 0, with d=2, w0=20.
"""
import deepxde as dde
import numpy as np

def analytical_oscillator(d, w0, t):
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    return np.exp(-d * t) * 2.0 * A * np.cos(phi + w * t)

def solve(iterations=4000, lr=1e-3):
    d, w0 = 2.0, 20.0
    mu, k = 2.0 * d, w0**2

    t_all = np.linspace(0.0, 1.0, 500)[:, None]
    u_all = analytical_oscillator(d, w0, t_all)

    # 10 data points on LHS
    t_data = t_all[0:200:20]
    u_data = u_all[0:200:20]

    geom = dde.geometry.TimeDomain(0.0, 1.0)

    def pde(x, y):
        dx = dde.grad.jacobian(y, x)
        dx2 = dde.grad.hessian(y, x)
        return dx2 + mu * dx + k * y

    observe = dde.icbc.PointSetBC(t_data, u_data)
    data = dde.data.TimePDE(geom, pde, [observe], num_domain=40, anchors=t_data)
    net = dde.nn.FNN([1] + [32] * 3 + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    # loss_weights: [PDE_residual (1e-4), PointSetBC (1.0)]
    model.compile("adam", lr=lr, loss_weights=[1e-4, 1.0])
    model.train(iterations=iterations, display_every=2000)

    u_pred = model.predict(t_all)
    rel_l2_err = float(np.linalg.norm(u_pred - u_all) / np.linalg.norm(u_all))
    return t_all, u_pred, u_all, rel_l2_err

if __name__ == "__main__":
    t, u_pred, u_all, rel_err = solve(iterations=4000)
    print(f"PINN p03: Harmonic DeepXDE PASSED (Relative L2 Error: {rel_err:.2%})")
    assert rel_err < 0.25, f"DeepXDE Harmonic error too high: {rel_err:.2%}"
