"""Step 06: 2D Nonlinear Convection (Coupled u, v Fields)
Equations:
  ∂u/∂t + u ∂u/∂x + v ∂u/∂y = 0
  ∂v/∂t + u ∂v/∂x + v ∂v/∂y = 0
"""
import numpy as np

def solve(nx=101, ny=101, nt=80, sigma=0.2):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = sigma * dx
    
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    u[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    v[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        u[1:, 1:] = (un[1:, 1:] - (un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                                  (vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
        v[1:, 1:] = (vn[1:, 1:] - (un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) -
                                  (vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])))
        u[0, :] = 1.0; u[-1, :] = 1.0; u[:, 0] = 1.0; u[:, -1] = 1.0
        v[0, :] = 1.0; v[-1, :] = 1.0; v[:, 0] = 1.0; v[:, -1] = 1.0
        
    return x, y, u, v

if __name__ == "__main__":
    x, y, u, v = solve()
    assert u.shape == (101, 101)
    print("Step 06: 2D Coupled Convection PASSED.")
