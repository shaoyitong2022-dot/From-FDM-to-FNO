# Style & Authoring Guide

This style guide establishes the pedagogical and engineering conventions for all courseware, code examples, and mathematical derivations in the **From FDM to FNO** project.

---

## 1. Mathematical Notation & Typography

1. **LaTeX Conventions**:
   - Time derivatives: $\partial u / \partial t$ or $\frac{\partial u}{\partial t}$.
   - Convective velocity vector: $\mathbf{u} = (u, v)$ in 2D, with bold vector notation.
   - Laplacians: $\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$.
   - Courant number: $\sigma = \frac{c \Delta t}{\Delta x} \le 1$.
   - Diffusion number: $r = \frac{\nu \Delta t}{\Delta x^2} \le \frac{1}{2}$.
2. **Bilingual Terminology**:
   - Upon first introduction in Chinese text, core technical terms must include their canonical English name: e.g. 迎风格式 (Upwind Scheme), 神经算子 (Neural Operator), 预条件子 (Preconditioner).
3. **Punctuation**:
   - Chinese prose uses full-width punctuation (`，`、`。`、`：`、`；`、`！`、`？`).
   - English sentences and mathematical code blocks use half-width standard punctuation.

---

## 2. Python Code Standards

1. **Self-Contained Execution**:
   - Code snippets inside lesson pages must be reproducible.
   - Top-level scripts must declare dependencies in their header docstrings.
2. **Explicit Memory Isolation**:
   - When updating time levels in finite difference schemes, always enforce explicit memory duplication:
     ```python
     # Correct:
     un = u.copy()
     # Prohibited:
     un = u  # Mutation alias bug
     ```
3. **Reproducibility**:
   - Pin all seeds before model initialization:
     ```python
     import torch
     import numpy as np

     torch.manual_seed(42)
     np.random.seed(42)
     ```

---

## 3. Lesson Design Components (Design System)

The CSS design system (`courseware/assets/course.css`) provides semantic visual containers:

- **`.box.prep`**: Pre-lesson prerequisites, cognitive goals, and physical setup.
- **`.box.mentor`**: First-principles physical intuition and Feynman inquiry.
- **`.box.deepen`**: Theoretical proofs, von Neumann amplification factor analyses.
- **`.win`**: Lesson takeaways, quantitative verification summary.
- **`details.quiz`**: Formative self-test retrieval cards.

---

## 4. The Scientific Closed-Loop Triad

Every computational problem must culminate in:
1. **One Diagnostic Plot**: Contour plot, streamlines, or logarithmic convergence curve.
2. **One Physical Principle**: Why this behavior occurs (dispersion vs. dissipation, CFL constraint, conservation of circulation).
3. **One Explicit Error Metric**: Relative $L_2$ norm vs. analytical solution or iteration count.
