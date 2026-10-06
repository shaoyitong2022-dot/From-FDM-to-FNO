"""Step 07: 2D Diffusion
Equation: ∂u/∂t = ν (∂²u/∂x² + ∂²u/∂y²)
Discretization: 2D Centered Space, Forward Time
"""
import numpy as np

def solve(nx=31, ny=31, nt=17, nu=0.05, sigma=0.25):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = sigma * dx * dy / nu
    
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx))
    u[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    
    for n in range(nt):
        un = u.copy()
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                         nu * dt / dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                         nu * dt / dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        u[0, :] = 1.0; u[-1, :] = 1.0; u[:, 0] = 1.0; u[:, -1] = 1.0
        
    return x, y, u

if __name__ == "__main__":
    x, y, u = solve()
    assert u.shape == (31, 31)
    print("Step 07: 2D Diffusion PASSED.")
