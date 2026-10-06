"""Step 03: 1D Diffusion Equation
Equation: ∂u/∂t = ν ∂²u/∂x²
Discretization: FTCS (Forward Time, Centered Space)
Verification: Analytical Sine Mode Decay exp(-ν k² t) * sin(k x)
"""
import numpy as np

def solve(nx=81, L=2.0, nu=0.1, T=0.5, dt=0.001):
    dx = L / (nx - 1)
    nt = int(round(T / dt))
    r = nu * dt / (dx ** 2)
    assert r <= 0.5, f"Diffusion stability condition r <= 0.5 violated: r = {r:.3f}"
    
    x = np.linspace(0.0, L, nx)
    k = 2.0 * np.pi / L
    u = np.sin(k * x)
    
    for _ in range(nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + r * (un[2:] - 2.0 * un[1:-1] + un[:-2])
        u[0] = un[0] + r * (un[1] - 2.0 * un[0] + un[-2])
        u[-1] = u[0]
        
    u_exact = np.exp(-nu * (k ** 2) * T) * np.sin(k * x)
    l2_err = np.sqrt(dx * np.sum((u - u_exact) ** 2))
    rel_err = l2_err / np.sqrt(dx * np.sum(u_exact ** 2))
    return x, u, u_exact, rel_err

if __name__ == "__main__":
    x, u, u_exact, rel_err = solve()
    assert rel_err < 1e-4, f"Diffusion error too large: {rel_err}"
    print(f"Step 03: Diffusion PASSED (Relative L2 Error: {rel_err:.2e})")
