"""
benchmarks/analytical_solutions.py
==================================
计算物理基准检验裁判库：提供经典偏微分方程与动力系统的闭式解析解（Analytical Closed-Form Solutions）
与经典文献基准数据（如 Ghia et al. 1982 腔流基准），为所有数值与神经网络解法提供零假设真值。
"""

import numpy as np
import sympy as sp


def linear_convection_exact(x: np.ndarray, t: float, c: float, domain_length: float = 2.0) -> np.ndarray:
    """一维线性对流方程方波平移真值（周期边界条件）"""
    # 考虑对流速度 c 的空间位移
    x_shift = (x - c * t) % domain_length
    u = np.ones_like(x)
    mask = (x_shift >= 0.5) & (x_shift <= 1.0)
    u[mask] = 2.0
    return u


def diffusion_sine_exact(x: np.ndarray, t: float, nu: float, L: float = 2.0) -> np.ndarray:
    """一维热传导/扩散方程正弦单模态精确衰减解（周期边界）
    方程：∂u/∂t = ν ∂²u/∂x²
    初值：u(x,0) = sin(2π x / L)
    解析解：u(x,t) = exp(-ν k² t) * sin(k x), k = 2π / L
    """
    k = 2.0 * np.pi / L
    return np.exp(-nu * (k**2) * t) * np.sin(k * x)


def burgers_cole_hopf_exact(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """一维黏性 Burgers 方程的 Cole-Hopf 变换闭式解析解
    方程：∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    边界：周期边界 [0, 2π]
    初值经 Cole-Hopf 反演构造得到：u = -2ν (∂φ/∂x)/φ + 4
    """
    x_sym, t_sym, nu_sym = sp.symbols('x t nu')
    phi = (sp.exp(-(x_sym - 4 * t_sym)**2 / (4 * nu_sym * (t_sym + 1))) +
           sp.exp(-(x_sym - 4 * t_sym - 2 * sp.pi)**2 / (4 * nu_sym * (t_sym + 1))))
    phiprime = phi.diff(x_sym)
    u_expr = -2 * nu_sym * (phiprime / phi) + 4
    ufunc = sp.lambdify((x_sym, t_sym, nu_sym), u_expr, 'numpy')
    return np.asarray(ufunc(x, t, nu))


def laplace_2d_series_exact(X: np.ndarray, Y: np.ndarray, n_terms: int = 40) -> np.ndarray:
    """二维 Laplace 方程 ∇²p = 0 的傅里叶级数解析解
    边界条件：
      p(0, y) = 0
      p(2, y) = y
      ∂p/∂y (x, 0) = 0
      ∂p/∂y (x, 1) = 0
    级数展开解：
      p(x, y) = x/4 - sum_{n=1,3,...} [4 / (n² π² sinh(2nπ))] * sinh(nπ x) * cos(nπ y)
    """
    p_exact = X / 4.0
    for n in range(1, n_terms, 2):
        n_pi = n * np.pi
        coeff = 4.0 / ((n_pi**2) * np.sinh(2.0 * n_pi))
        term = coeff * np.sinh(n_pi * X) * np.cos(n_pi * Y)
        p_exact -= term
    return p_exact


def poisson_1d_exact(x: np.ndarray) -> np.ndarray:
    """一维泊松方程 -u''(x) = 1, u(0)=u(1)=0 的解析解
    解析式：u(x) = 0.5 * x * (1 - x)
    """
    return 0.5 * x * (1.0 - x)


def harmonic_oscillator_exact(t: np.ndarray, d: float = 2.0, w0: float = 20.0) -> np.ndarray:
    """二阶欠阻尼谐振子 ODE 解析解
    方程：d²x/dt² + 2d dx/dt + w0² x = 0, x(0)=1, x'(0)=0
    """
    assert d < w0, "Only underdamped case (d < w0) supported"
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1.0 / (2.0 * np.cos(phi))
    return np.exp(-d * t) * 2.0 * A * np.cos(phi + w * t)


def ghia_cavity_re100_benchmark():
    """Ghia, Ghia and Shin (1982) Lid-driven cavity benchmark data for Re=100
    返回竖直中心线 (x=0.5) 上的 u 速度分布点 (y, u)
    """
    y = np.array([
        0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
        0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000
    ])
    u = np.array([
        0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
        -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000
    ])
    return y, u
