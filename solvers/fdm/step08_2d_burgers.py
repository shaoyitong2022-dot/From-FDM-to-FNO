"""Step 08: 2D Burgers' Equation
Equations:
  ∂u/∂t + u ∂u/∂x + v ∂u/∂y = ν ∇²u
  ∂v/∂t + u ∂v/∂x + v ∂v/∂y = ν ∇²v
"""
import numpy as np

def solve(nx=41, ny=41, nt=120, nu=0.01, sigma=0.0009):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = sigma * dx * dy / nu
    
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    u[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    v[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                         nu * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                         nu * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                         nu * dt / dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                         nu * dt / dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
        u[0, :] = 1.0; u[-1, :] = 1.0; u[:, 0] = 1.0; u[:, -1] = 1.0
        v[0, :] = 1.0; v[-1, :] = 1.0; v[:, 0] = 1.0; v[:, -1] = 1.0
        
    return x, y, u, v

if __name__ == "__main__":
    x, y, u, v = solve()
    assert u.shape == (41, 41)
    print("Step 08: 2D Burgers PASSED.")
