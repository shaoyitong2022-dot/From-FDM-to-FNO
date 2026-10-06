# Contributing to From FDM to FNO

Thank you for your interest in contributing to the **From FDM to FNO** suite. We welcome contributions that uphold the scientific rigor, reproducibility, and pedagogical clarity of computational physics and SciML.

---

## 1. Guiding Principles

1. **Rigorous Verification Over Ad-Hoc Scripts**:
   Every added solver or numerical example must be paired with:
   - An exact or asymptotic analytical solution, or
   - An established peer-reviewed benchmark (e.g., Ghia 1982, Cole-Hopf), or
   - An empirical convergence rate test ($p$-order verification).
2. **Deterministic & Self-Contained**:
   - Random seeds must be explicitly pinned (`torch.manual_seed(42)`, `np.random.seed(42)`).
   - Scripts in `solvers/` must be standalone runnable with standard scientific dependencies.
3. **Delivery Standard**:
   Every computational experiment must satisfy the **Closed-Loop Triad**:
   - One diagnostic plot / visualization,
   - One concise physical conclusion,
   - One explicit numerical error metric (e.g., relative $L_2$ error, iteration count).

---

## 2. Types of Contributions We Welcome

- **Bug Reports & Accuracy Fixes**: Identifying numerical discrepancies, edge-case instabilities, or typographical errors in equations and code.
- **Upstream Framework Maintenance**: Keeping compatibility with newest PyTorch, DeepXDE, and NeuralOperator releases.
- **Benchmark Extensions**: Adding verified physical test cases (e.g., Sod shock tube, lid-driven cavity at higher Reynolds numbers).
- **Documentation & Clarifications**: Enhancing the mathematical derivations, physical intuition, and bilingual glossaries.

---

## 3. Pull Request (PR) Workflow

1. **Fork & Branch**:
   Create a focused feature branch from `main`:
   ```bash
   git checkout -b feat/fno-boundary-condition
   ```
2. **Run Local Regression Suite**:
   Before submitting your PR, ensure all 20 solver kernels and unit tests pass:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   python benchmarks/convergence_tests.py
   python benchmarks/cfl_stability_scan.py
   python benchmarks/run_all_benchmarks.py
   ```
3. **Commit Standards**:
   Use standard conventional commits:
   - `fix(fdm): correct pressure Poisson Neumann boundary indexing`
   - `feat(pinn): add analytical comparison for damped oscillator`
   - `docs(fno): refine green function continuous kernel derivation`
4. **Open a Pull Request**:
   Fill in the PR template detailing the physical context, numerical verification, and wall-time impact.
