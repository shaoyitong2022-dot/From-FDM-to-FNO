"""Step 05: 2D Linear Convection
Equation: ∂u/∂t + c ∂u/∂x + c ∂u/∂y = 0
Discretization: 2D Upwind (Backward Difference in both x and y)
"""
import numpy as np

def solve(nx=81, ny=81, nt=100, c=1.0, sigma=0.2):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    dt = sigma * dx
    
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx))
    u[int(0.5/dy):int(1.0/dy+1), int(0.5/dx):int(1.0/dx+1)] = 2.0
    
    for n in range(nt):
        un = u.copy()
        u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                                  (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
        u[0, :] = 1.0; u[-1, :] = 1.0; u[:, 0] = 1.0; u[:, -1] = 1.0
        
    return x, y, u

if __name__ == "__main__":
    x, y, u = solve()
    assert u.shape == (81, 81)
    assert 1.0 <= np.max(u) <= 2.01
    print("Step 05: 2D Linear Convection PASSED.")
