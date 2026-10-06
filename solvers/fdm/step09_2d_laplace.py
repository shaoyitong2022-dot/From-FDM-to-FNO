"""Step 09: 2D Laplace Equation
Equation: ∇²p = ∂²p/∂x² + ∂²p/∂y² = 0
Discretization: 5-Point Stencil Relaxation
Verification: Analytical Fourier Series Solution
"""
import numpy as np

def solve(nx=31, ny=31, tol=1e-4, max_iter=3000):
    dx = 2.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    p = np.zeros((ny, nx))
    p[:, -1] = y  # 边界: p = y @ x = 2
    
    for it in range(1, max_iter + 1):
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) +
                          dx**2 * (pn[2:, 1:-1] + pn[:-2, 1:-1])) /
                         (2.0 * (dx**2 + dy**2)))
        p[:, 0] = 0.0          # x = 0
        p[:, -1] = y           # x = 2
        p[0, :] = p[1, :]      # ∂p/∂y = 0 @ y = 0
        p[-1, :] = p[-2, :]    # ∂p/∂y = 0 @ y = 1
        
        diff = np.sum(np.abs(p - pn)) / (np.sum(np.abs(p)) + 1e-12)
        if diff < tol:
            break
            
    # 解析解 (傅里叶级数)
    p_exact = X / 4.0
    for n in range(1, 40, 2):
        term = (4.0 / ((n * np.pi)**2 * np.sinh(2.0 * n * np.pi))) * np.sinh(n * np.pi * X) * np.cos(n * np.pi * Y)
        p_exact -= term
        
    l2_err = np.sqrt(dx * dy * np.sum((p - p_exact)**2))
    rel_err = l2_err / np.sqrt(dx * dy * np.sum(p_exact**2))
    return x, y, p, p_exact, rel_err, it

if __name__ == "__main__":
    x, y, p, p_exact, rel_err, it = solve()
    assert rel_err < 0.10, f"Laplace relative error too high: {rel_err}"
    print(f"Step 09: 2D Laplace PASSED (Converged in {it} iters, Rel L2 Error: {rel_err:.2%})")
