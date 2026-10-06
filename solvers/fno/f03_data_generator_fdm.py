"""f03_data_generator_fdm.py: Bridge FDM simulation to FNO dataset.
Generates 1D Viscous Burgers operator dataset u_0(x) -> u(x, T) using vectorized FDM.
Produces PyTorch dataset ready for NeuralOperator training.
"""
import torch
import numpy as np

def burgers_fdm_periodic(u0: np.ndarray, nu=0.01, T=0.5) -> np.ndarray:
    """Vectorized 1D Burgers solver with sign-aware upwind & periodic boundary conditions.
    Equation: du/dt + u * du/dx = nu * d^2u/dx^2
    Sign-aware upwind:
      When u >= 0: backward difference (un - un_left)/dx
      When u < 0:  forward difference (un_right - un)/dx
    """
    n = len(u0)
    dx = 1.0 / n
    max_u = max(float(np.max(np.abs(u0))), 1e-4)
    dt = 0.25 * min(dx / max_u, 0.5 * (dx ** 2) / nu)
    nt = int(T / dt)
    u = u0.astype(np.float64).copy()

    for _ in range(nt):
        un = u.copy()
        du_b = (un - np.roll(un, 1)) / dx
        du_f = (np.roll(un, -1) - un) / dx
        adv = np.maximum(un, 0.0) * du_b + np.minimum(un, 0.0) * du_f
        d2u_dx2 = (np.roll(un, -1) - 2.0 * un + np.roll(un, 1)) / (dx ** 2)
        u = un - dt * adv + dt * nu * d2u_dx2

    return u

def generate_operator_dataset(num_samples=100, nx=64, nu=0.01, T=0.5, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    X_list, Y_list = [], []

    for _ in range(num_samples):
        # Multi-modal smooth initial condition
        u0 = np.zeros_like(x)
        for k in range(1, 5):
            a, b = rng.randn(2) / k
            u0 += a * np.sin(2.0 * np.pi * k * x) + b * np.cos(2.0 * np.pi * k * x)
        u0 = u0 / np.max(np.abs(u0))  # normalize max amplitude to 1.0
        uT = burgers_fdm_periodic(u0, nu=nu, T=T)

        X_list.append(u0)
        Y_list.append(uT)

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32).unsqueeze(1)
    return X_tensor, Y_tensor

if __name__ == "__main__":
    X, Y = generate_operator_dataset(num_samples=20, nx=64)
    assert X.shape == (20, 1, 64)
    assert Y.shape == (20, 1, 64)
    # Energy dissipation check: int(u^2) should decrease due to viscosity nu
    e0 = float(torch.mean(X ** 2))
    eT = float(torch.mean(Y ** 2))
    print(f"FNO f03: Data Generator PASSED (E(0)={e0:.4f} -> E(T)={eT:.4f}, Energy Dissipation: {((e0-eT)/e0):.2%})")
    assert eT < e0, "Viscous dissipation violated!"
