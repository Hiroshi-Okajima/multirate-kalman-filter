# LMI-Based Multirate Kalman Filter Design

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)](https://www.mathworks.com/products/matlab.html)

MATLAB implementation of LMI-based multirate Kalman filter design using cyclic reformulation. This repository contains the code accompanying the paper:

> H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter Design," *IEEE Access*, 2025. [[Preprint: arXiv:XXXX.XXXXX]](https://arxiv.org/abs/XXXX.XXXXX)

### Main Scripts

| File | Description | Design Objective |
|------|-------------|------------------|
| `MultirateKF_01.m` | Basic optimal Kalman filter design | Minimize trace(P_e) |
| `MultirateKF_02_eig.m` | Multi-objective design with eigenvalue placement | Minimize trace(P_e) subject to \|λ\| < r̄ |
| `MultirateKF_03_l2.m` | Multi-objective design with l2-induced norm | Minimize trace(P_e) subject to \|\|G\|\|_{l2} < γ̄ |

### Example System

Automotive navigation with GPS (1 Hz) and wheel speed sensor (10 Hz):
- **State**: [position; velocity; acceleration]
- **Period**: N = 10 steps
- **Measurements**: 
  - k mod 10 = 0: GPS + wheel speed
  - k mod 10 ≠ 0: wheel speed only
