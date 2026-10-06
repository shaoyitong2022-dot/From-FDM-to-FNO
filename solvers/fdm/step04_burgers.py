"""Step 04: 1D Burgers' Equation
Equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
Discretization: Upwind convection + Centered diffusion
Verification: Cole-Hopf Analytical Solution
"""
import numpy as np
import sympy as sp

def solve(nx=101, nt=100, nu=0.07):
    L = 2.0 * np.pi
    dx = L / (nx - 1)
    dt = dx * nu
    T = nt * dt
    x = np.linspace(0.0, L, nx)
    
    # SymPy Cole-Hopf 符号解析解
    x_s, t_s, nu_s = sp.symbols('x t nu')
    phi = (sp.exp(-(x_s - 4*t_s)**2 / (4*nu_s*(t_s + 1))) +
           sp.exp(-(x_s - 4*t_s - 2*sp.pi)**2 / (4*nu_s*(t_s + 1))))
    u_sym = -2*nu_s*(phi.diff(x_s) / phi) + 4
    ufunc = sp.lambdify((t_s, x_s, nu_s), u_sym, 'numpy')
    
    u = np.asarray([ufunc(0.0, xi, nu) for xi in x])
    u_exact = np.asarray([ufunc(T, xi, nu) for xi in x])
    
    for _ in range(nt):
        un = u.copy()
        # 内部点：迎风对流 + 中心扩散
        u[1:-1] = (un[1:-1] - un[1:-1] * (dt / dx) * (un[1:-1] - un[:-2]) +
                   nu * (dt / dx**2) * (un[2:] - 2*un[1:-1] + un[:-2]))
        # 周期边界
        u[0] = (un[0] - un[0] * (dt / dx) * (un[0] - un[-2]) +
                nu * (dt / dx**2) * (un[1] - 2*un[0] + un[-2]))
        u[-1] = u[0]
        
    l2_err = np.sqrt(dx * np.sum((u - u_exact)**2))
    rel_err = l2_err / np.sqrt(dx * np.sum(u_exact**2))
    return x, u, u_exact, rel_err

if __name__ == "__main__":
    x, u, u_exact, rel_err = solve()
    assert rel_err < 0.20, f"Burgers error too large: {rel_err}"
    print(f"Step 04: Burgers PASSED (Relative L2 Error: {rel_err:.2%})")
