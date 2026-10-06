"""Step 01: 1D Linear Convection
Equation: ∂u/∂t + c ∂u/∂x = 0
Discretization: Forward Time, Backward Space (FTBS / 1st-order Upwind)
"""
import numpy as np

def solve(nx=81, L=2.0, c=1.0, nt=25, dt=0.02):
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u = np.ones(nx)
    u[int(0.5 / dx):int(1.0 / dx + 1)] = 2.0  # 方波初值
    
    sigma = c * dt / dx
    assert sigma <= 1.0, f"CFL violated: sigma = {sigma:.3f} > 1.0"
    
    for n in range(nt):
        un = u.copy()
        u[1:] = un[1:] - sigma * (un[1:] - un[:-1])
        u[0] = 1.0
        
    return x, u

if __name__ == "__main__":
    x, u = solve()
    assert len(u) == 81
    assert 1.0 <= np.max(u) <= 2.01
    print("Step 01: Linear Convection PASSED.")
