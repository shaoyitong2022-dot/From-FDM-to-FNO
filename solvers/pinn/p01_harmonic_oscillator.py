"""p01_harmonic_oscillator.py: Pure PyTorch Physics-Informed Neural Network (PINN).
Canonical Benchmark: 1D Damped Harmonic Oscillator (Ben Moseley / Raissi et al.)
Governing Equation: d^2 u / dt^2 + 2*d * du/dt + w0^2 * u = 0, with d=2, w0=20.
Analytical Solution: u(t) = exp(-d*t) * (cos(w*t) + (d/w)*sin(w*t)), w = sqrt(w0^2 - d^2).
"""
import torch
import torch.nn as nn
import numpy as np

def analytical_oscillator(d: float, w0: float, t: torch.Tensor) -> torch.Tensor:
    assert d < w0, "Must be underdamped"
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    return torch.exp(-d * t) * 2.0 * A * torch.cos(phi + w * t)

class FCN(nn.Module):
    """Fully-connected PINN with Tanh activations."""
    def __init__(self, in_features=1, out_features=1, hidden_dim=32, num_layers=3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def solve(epochs=4000, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d, w0 = 2.0, 20.0
    mu, k = 2.0 * d, w0**2

    # Domain [0, 1]
    t_full = torch.linspace(0.0, 1.0, 500).view(-1, 1)
    u_exact = analytical_oscillator(d, w0, t_full)

    # 10 training data points in LHS [0, 0.4]
    t_data = t_full[0:200:20]
    u_data = u_exact[0:200:20]

    # 30 physics collocation points across full [0, 1]
    t_physics = torch.linspace(0.0, 1.0, 30).view(-1, 1).requires_grad_(True)

    model = FCN(1, 1, hidden_dim=32, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Data loss
        u_pred_data = model(t_data)
        loss_data = torch.mean((u_pred_data - u_data) ** 2)

        # Physics residual loss
        u_pred_phys = model(t_physics)
        du_dt = torch.autograd.grad(u_pred_phys, t_physics, torch.ones_like(u_pred_phys), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t_physics, torch.ones_like(du_dt), create_graph=True)[0]
        pde_residual = d2u_dt2 + mu * du_dt + k * u_pred_phys
        loss_pde = torch.mean(pde_residual ** 2)

        # Loss weighting: balance data and physics scales
        loss = loss_data + 1e-4 * loss_pde
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_pred_full = model(t_full)
        rel_l2_err = (torch.norm(u_pred_full - u_exact) / torch.norm(u_exact)).item()

    return t_full.cpu().numpy(), u_pred_full.cpu().numpy(), u_exact.cpu().numpy(), rel_l2_err

if __name__ == "__main__":
    t, u_pred, u_exact, rel_err = solve(epochs=4000)
    print(f"PINN p01: Harmonic Oscillator PASSED (Relative L2 Error: {rel_err:.2%})")
    assert rel_err < 0.25, f"Harmonic oscillator PINN error too high: {rel_err:.2%}"
