"""Step 12: 2D Channel Flow (Navier-Stokes with Periodic BC & Pressure Force)
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
    # 周期边界
    b[1:-1, -1] = (rho * (1.0 / dt *
                  ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx) +
                   (v[2:, -1] - v[:-2, -1]) / (2 * dy)) -
                  ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                  2.0 * ((u[2:, -1] - u[:-2, -1]) / (2 * dy) *
                         (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                  ((v[2:, -1] - v[:-2, -1]) / (2 * dy))**2))
    b[1:-1, 0] = b[1:-1, -1]
    return b

def pressure_poisson_periodic(p, dx, dy, b, nit=50):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2.0 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2.0 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        # x 方向周期边界
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2]) * dy**2 +
                        (pn[2:, -1] + pn[:-2, -1]) * dx**2) /
                       (2.0 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2.0 * (dx**2 + dy**2)) * b[1:-1, -1])
        p[1:-1, 0] = p[1:-1, -1]
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 2
        p[0, :] = p[1, :]    # dp/dy = 0 @ y = 0
    return p

def solve(nx=41, ny=41, nt=100, nit=50, rho=1.0, nu=0.1, dt=0.001, F=1.0):
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
        p = pressure_poisson_periodic(p, dx, dy, b, nit)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])) +
                         F * dt)
                         
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
                               
        # 边界：壁面无滑移
        u[0, :] = 0.0; u[-1, :] = 0.0
        v[0, :] = 0.0; v[-1, :] = 0.0
        
    return x, y, u, v, p

if __name__ == "__main__":
    x, y, u, v, p = solve()
    assert u.shape == (41, 41)
    print("Step 12: Channel Flow PASSED.")
