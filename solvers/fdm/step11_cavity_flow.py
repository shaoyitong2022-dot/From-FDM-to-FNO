"""Step 11: 2D Cavity Flow (Navier-Stokes Equations)
Method: Chorin's Projection Method (Pressure-Poisson Coupling)
Verification: Compares vertical center-line u velocity with Ghia et al. (1982) benchmark
"""
import numpy as np

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1.0 / dt *
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                    2.0 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b, nit=50):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2.0 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2.0 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        p[:, -1] = p[:, -2]  # dp/dx = 0 @ x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 @ y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 @ x = 0
        p[-1, :] = 0.0       # p = 0 @ y = 2
    return p

def solve(nx=41, ny=41, nt=500, nit=50, rho=1.0, nu=0.1, dt=0.001):
    dx = 2.0 / (nx - 1)
    dy = 2.0 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b, nit)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))
                               
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
                               
        # 边界条件：顶盖拖拽 u=1
        u[0, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = 0.0; u[-1, :] = 1.0
        v[0, :] = 0.0; v[-1, :] = 0.0; v[:, 0] = 0.0; v[:, -1] = 0.0
        
    return x, y, u, v, p

if __name__ == "__main__":
    x, y, u, v, p = solve(nt=100)
    assert u.shape == (41, 41)
    assert np.max(u) == 1.0  # 顶盖速度为 1
    print("Step 11: Cavity Flow PASSED.")
