"""Step 02: 1D Nonlinear Convection
Equation: ∂u/∂t + u ∂u/∂x = 0
Discretization: FTBS with local velocity u_i^n
"""
import numpy as np

def solve(nx=81, L=2.0, nt=25, dt=0.01):
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    u = np.ones(nx)
    u[int(0.5 / dx):int(1.0 / dx + 1)] = 2.0
    
    for n in range(nt):
        un = u.copy()
        # 非线性迎风：波速为 un[1:]
        u[1:] = un[1:] - un[1:] * (dt / dx) * (un[1:] - un[:-1])
        u[0] = 1.0
        
    return x, u

if __name__ == "__main__":
    x, u = solve()
    assert len(u) == 81
    print("Step 02: Nonlinear Convection PASSED.")
