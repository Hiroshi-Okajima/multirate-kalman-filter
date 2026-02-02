# LMI-Based Multirate Kalman Filter Design

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)](https://www.mathworks.com/products/matlab.html)

MATLAB implementation of LMI-based multirate Kalman filter design using cyclic reformulation. This repository contains the code accompanying the paper:

> H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter Design," *IEEE Access*, 2025. [[Preprint: arXiv:XXXX.XXXXX]](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

In multirate systems, sensors operate at different sampling rates, leading to periodic time-varying measurement availability. This repository provides:

- **Optimal Kalman filter design** via LMI optimization (DARI)
- **Multi-objective design** with eigenvalue placement constraints
- **Robust design** with l2-induced norm constraints
- **Cyclic reformulation** that naturally handles semidefinite measurement noise covariance

### Key Features

- ✅ Handles semidefinite measurement noise covariance `R_cyc ≽ 0` (not positive definite)
- ✅ Unified LMI framework for multiple design objectives
- ✅ Periodic steady-state Kalman gains computed offline
- ✅ Trade-off analysis between average performance and robustness/convergence rate

## System Model

### Multirate System
```matlab
x(k+1) = A*x(k) + B*u(k) + w(k)        % State dynamics
y(k)   = S_k*C*x(k) + S_k*v(k)         % Periodic measurements
```

- `S_k`: Diagonal selection matrix indicating sensor availability
- `S_{k+N} = S_k`: Periodic with period N

### Cyclic Reformulation
The periodic system is transformed into a time-invariant cyclic system:
```matlab
x̌(k+1) = Ǎ*x̌(k) + B̌*u(k) + w̌(k)
y̌(k)   = Č*x̌(k) + v̌(k)
```

where `Ǎ`, `Č`, `Q̌`, `Ř` have special cyclic/block-diagonal structures.

## Files

### Main Scripts

| File | Description | Design Objective |
|------|-------------|------------------|
| `MultirateKF_LMI_Rcyc_success_02.m` | Basic optimal Kalman filter design | Minimize trace(P_e) |
| `MultirateKF_02_eig.m` | Multi-objective design with eigenvalue placement | Minimize trace(P_e) subject to \|λ\| < r̄ |
| `MultirateKF_03_l2.m` | Multi-objective design with l2-induced norm | Minimize trace(P_e) subject to \|\|G\|\|_{l2} < γ̄ |

### Example System

Automotive navigation with GPS (1 Hz) and wheel speed sensor (10 Hz):
- **State**: [position; velocity; acceleration]
- **Period**: N = 10 steps
- **Measurements**: 
  - k mod 10 = 0: GPS + wheel speed
  - k mod 10 ≠ 0: wheel speed only

## Usage

### 1. Optimal Kalman Filter Design
```matlab
% Run basic optimal design
MultirateKF_LMI_Rcyc_success_02

% Output:
% - Optimal Kalman gains L_k (k=0,...,9)
% - Estimation error covariance trace(P_e)
% - Closed-loop eigenvalues
% - Simulation results with RMSE
```

**Key Results:**
- Position RMSE: ~0.600 m
- Velocity RMSE: ~0.268 m/s
- Max eigenvalue magnitude: ~0.967

### 2. Design with Eigenvalue Constraints
```matlab
% Run design with convergence rate constraint
MultirateKF_02_eig

% Trade-off analysis: r̄ from 0.975 to 0.75
% Output:
% - Trade-off curve: trace(W) vs r̄
% - Faster convergence → larger trace(W)
```

**Design Choice:**
- Relaxed (r̄ ≈ 0.9): Small performance penalty, moderate convergence
- Strict (r̄ ≈ 0.7): Large performance penalty, fast convergence

### 3. Design with l2-Induced Norm Constraint
```matlab
% Run design with worst-case performance constraint
MultirateKF_03_l2

% Trade-off analysis: γ̄ from 10×γ_opt to 1.01×γ_opt
% Output:
% - Trade-off curve: trace(W) vs γ̄/γ_opt
% - Better robustness → larger trace(W)
```

**Design Choice:**
- Relaxed (γ̄ ≫ γ_opt): Near-optimal average performance
- Strict (γ̄ ≈ γ_opt): Better worst-case performance, degraded average

## Theoretical Background

### LMI Formulation

Unified block matrix form:
```
[X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
[*,    (2,2),   0,         0       ]
[*,    0,       (3,3),     0       ]  ≻ 0
[*,    0,       0,         (4,4)   ]
```

| Design | (2,2) | (3,3) | (4,4) | Objective |
|--------|-------|-------|-------|-----------|
| Optimal Kalman | X | I | I | min trace(W) where W ≽ X^{-1} |
| Eigenvalue | r̄²X | I | I | min trace(W) s.t. \|λ\| < r̄ |
| l2-induced | X-I | γ̄²I | γ̄²I | min trace(W) s.t. \|\|G\|\| < γ̄ |

- **Variables**: X (Lyapunov matrix, X = P_e^{-1}), Y = -XL (gain)
- **Gain recovery**: L = -X^{-1}Y

### Why LMI?

Standard DARE requires `R̃ ≻ 0` (positive definite):
```matlab
P = APA^T - APC^T(CPC^T + R)^{-1}CPA^T + Q
```

In multirate systems with intermittent sensors:
```matlab
R̃_cyc = diag(S_0*R*S_0^T, ..., S_{N-1}*R*S_{N-1}^T)
```
- When S_k has zero rows → R̃_cyc is **semidefinite** ✗
- Standard DARE solvers fail or become numerically unstable

LMI approach via dual LQR:
- ✅ Naturally handles R̃_cyc ≽ 0
- ✅ Convex optimization (global optimum guaranteed)
- ✅ Easy to add multiple constraints

## Requirements

### MATLAB Toolboxes
- **Robust Control Toolbox** (for LMI optimization)
- **Control System Toolbox** (for basic system analysis)

### Tested Environment
- MATLAB R2020b or later
- Windows 10/11, macOS, Linux

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/multirate-kalman-filter.git
cd multirate-kalman-filter

# Run in MATLAB
matlab
>> MultirateKF_LMI_Rcyc_success_02
```

## Results

### Estimation Performance

| State | RMSE | Max Error |
|-------|------|-----------|
| Position | 0.600 m | ~2 m |
| Velocity | 0.268 m/s | ~1 m/s |
| Acceleration | 1.165 m/s² | ~5 m/s² |

### Trade-off Analysis

**Eigenvalue Placement:**
- r̄ = 0.975: trace(W) = 19.64 (8% increase)
- r̄ = 0.900: trace(W) = 41.19 (128% increase)
- r̄ = 0.750: trace(W) = 422.1 (2237% increase)

**l2-Induced Norm:**
- γ̄ = 10×γ_opt: trace(W) = 18.71 (3% increase)
- γ̄ = 1.5×γ_opt: trace(W) = 23.08 (28% increase)
- γ̄ = 1.01×γ_opt: trace(W) = 34.65 (92% increase)

## Citation

If you use this code in your research, please cite:
```bibtex
@article{okajima2025multirate,
  title={LMI Optimization Based Multirate Steady-State Kalman Filter Design},
  author={Okajima, Hiroshi},
  journal={IEEE Access},
  year={2025},
  note={Preprint: arXiv:XXXX.XXXXX}
}
```

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit and indicate if changes were made

## Author

**Hiroshi Okajima**  
Graduate School of Science and Technology, Kumamoto University, Japan  
Email: okajima@cs.kumamoto-u.ac.jp

## Acknowledgments

This work was created with assistance from Claude (Anthropic) for:
- Improving readability and language clarity
- Technical verification of mathematical expressions
- Code documentation and comments

All technical content, research methodology, results, and conclusions are entirely the author's own work.

## Related Work

- [Cyclic System Analysis](https://doi.org/10.1016/S0005-1098(00)00094-4) - Bittanti & Colaneri, Automatica, 2000
- [Periodic Kalman Filtering](https://ieeexplore.ieee.org/document/7527050) - Fujimoto et al., IEEE CCA, 2016
- [l2-Induced Norm for Periodic Systems](https://doi.org/10.1109/ACCESS.2023.3252547) - Okajima et al., IEEE Access, 2023

## Issues and Contributions

If you find any bugs or have suggestions for improvements:
- Open an issue on GitHub
- Submit a pull request

Please note that this is research code provided as-is for reproducibility purposes.

---

**Note**: Replace `XXXX.XXXXX` with actual arXiv number and update GitHub username in clone URL once published.
