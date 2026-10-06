"""Step 10: 2D Poisson Equation
Equation: ∇²p = b
Discretization: 5-Point Stencil Relaxation with symmetric point sources
"""
import numpy as np

def solve(nx=50, ny=50, nt=100):
    xmin, xmax, ymin, ymax = 0.0, 2.0, 0.0, 1.0
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    # 对称双尖峰源项
    b[int(ny / 4), int(nx / 4)] = 100.0
    b[int(3 * ny / 4), int(3 * nx / 4)] = -100.0
    
    for it in range(nt):
        pd = p.copy()
        p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
                          (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
                          b[1:-1, 1:-1] * dx**2 * dy**2) /
                         (2.0 * (dx**2 + dy**2)))
        p[0, :] = 0.0; p[-1, :] = 0.0; p[:, 0] = 0.0; p[:, -1] = 0.0
        
    return x, y, p, b

if __name__ == "__main__":
    x, y, p, b = solve()
    assert p.shape == (50, 50)
    assert np.max(p) > 0 and np.min(p) < 0
    print("Step 10: 2D Poisson PASSED.")
