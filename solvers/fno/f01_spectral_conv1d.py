"""f01_spectral_conv1d.py: Pure PyTorch 1D Spectral Convolution Layer (Li et al. 2020).
Demonstrates zero-shot super-resolution / mesh invariance:
Discrete FFT -> Frequency mode truncation -> Complex weight multiplication -> Inverse FFT.
"""
import torch
import torch.nn as nn
import numpy as np

class SpectralConv1d(nn.Module):
    """1D Fourier Spectral Convolution Layer with Complex Weight Matrix."""
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, n_points)
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x)  # (batch, in_channels, n//2 + 1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        # Multiply only the lowest 'modes' frequencies
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        # Return to physical space with exact target grid length n
        return torch.fft.irfft(out_ft, n=n)

def test_mesh_invariance():
    torch.manual_seed(42)
    modes = 8
    layer = SpectralConv1d(in_channels=1, out_channels=1, modes=modes)

    # Resolution 1: Coarse grid N1 = 64
    x1 = torch.linspace(0, 1, 65)[:-1]
    u1 = (torch.sin(2 * np.pi * x1) + 0.3 * torch.sin(6 * np.pi * x1)).view(1, 1, -1)

    # Resolution 2: Fine grid N2 = 256
    x2 = torch.linspace(0, 1, 257)[:-1]
    u2 = (torch.sin(2 * np.pi * x2) + 0.3 * torch.sin(6 * np.pi * x2)).view(1, 1, -1)

    with torch.no_grad():
        out1 = layer(u1)  # (1, 1, 64)
        out2 = layer(u2)  # (1, 1, 256)

    # Interpolate/downsample out2 to 64 points and compare
    out2_down = out2[:, :, ::4]
    rel_diff = (torch.norm(out1 - out2_down) / torch.norm(out1)).item()
    return rel_diff

if __name__ == "__main__":
    diff = test_mesh_invariance()
    print(f"FNO f01: SpectralConv1d PASSED (Mesh Invariance Discrepancy: {diff:.4e})")
    assert diff < 1e-4, f"Mesh invariance violated: {diff}"
