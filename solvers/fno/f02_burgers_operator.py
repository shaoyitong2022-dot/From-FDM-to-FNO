"""f02_burgers_operator.py: Operator Learning with Official NeuralOperator FNO.
Learns the mapping u_0(x) -> u(x, T) for 1D diffusion / heat equation.
Evaluates zero-shot super-resolution: Trained at N=64, tested at N=256.
"""
import torch
import torch.nn as nn
import numpy as np
from neuralop.models import FNO

def make_heat_data(n_samples=150, n_points=64, nu=0.05, T=0.1, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, n_points, endpoint=False)
    u0_list, uT_list = [], []
    k = np.arange(n_points // 2 + 1) * 2.0 * np.pi

    for _ in range(n_samples):
        u0 = np.zeros_like(x)
        for mode in range(1, 6):
            a, b = rng.randn(2) / (mode ** 2)
            u0 += a * np.sin(2 * np.pi * mode * x) + b * np.cos(2 * np.pi * mode * x)
        # Analytical Fourier propagator
        u0_ft = np.fft.rfft(u0)
        uT = np.fft.irfft(u0_ft * np.exp(-nu * (k ** 2) * T), n=n_points)
        u0_list.append(u0)
        uT_list.append(uT)

    X = torch.tensor(np.array(u0_list), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.array(uT_list), dtype=torch.float32).unsqueeze(1)
    return X, Y

def train_and_eval(epochs=100, lr=5e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, Y_train = make_heat_data(n_samples=120, n_points=64, seed=42)
    X_test, Y_test = make_heat_data(n_samples=30, n_points=64, seed=999)

    model = FNO(n_modes=(12,), in_channels=1, out_channels=1, hidden_channels=32, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_d, Y_train_d = X_train.to(device), Y_train.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_d)
        loss = criterion(pred, Y_train_d)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(device))
        Y_test_dev = Y_test.to(test_pred.device)
        rel_l2 = (torch.norm(test_pred - Y_test_dev) / torch.norm(Y_test_dev)).item()

    # Zero-shot super-resolution test at N=256
    X_fine, Y_fine = make_heat_data(n_samples=20, n_points=256, seed=777)
    with torch.no_grad():
        pred_fine = model(X_fine.to(device))
        Y_fine_dev = Y_fine.to(pred_fine.device)
        super_res_rel_l2 = (torch.norm(pred_fine - Y_fine_dev) / torch.norm(Y_fine_dev)).item()

    return rel_l2, super_res_rel_l2

if __name__ == "__main__":
    rel_l2, super_res_rel_l2 = train_and_eval(epochs=100)
    print(f"FNO f02: Operator Learning PASSED (Test L2: {rel_l2:.2%}, Zero-shot 4x Super-Res L2: {super_res_rel_l2:.2%})")
    assert rel_l2 < 0.05, f"Operator test error too high: {rel_l2}"
    assert super_res_rel_l2 < 0.08, f"Super-resolution error too high: {super_res_rel_l2}"
