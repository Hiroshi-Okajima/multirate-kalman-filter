# LMI-Based Multirate Kalman Filter Design

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)](https://www.mathworks.com/products/matlab.html)

MATLAB implementation of LMI-based multirate Kalman filter design using cyclic reformulation. This repository contains the code accompanying the paper:

> H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter Design,"
> ArXiV, Submitted to IEEE Access

### Main Scripts

| File | Description | Design Objective |
|------|-------------|------------------|
| `MultirateKF_01.m` | Basic optimal Kalman filter design | Minimize trace(P_e) |
| `MultirateKF_02_eig.m` | Multi-objective design with eigenvalue placement | Minimize trace(P_e) subject to \|λ\| < r̄ |
| `MultirateKF_03_l2.m` | Multi-objective design with l2-induced norm | Minimize trace(P_e) subject to \|\|G\|\|_{l2} < γ̄ |

#### Required 
Control System Toolbox
Robust Control Toolbox

### Example System

Automotive navigation with GPS (1 Hz) and wheel speed sensor (10 Hz):
- **State**: [position; velocity; acceleration]
- **Period**: N = 10 steps
- **Measurements**: 
  - k mod 10 = 0: GPS + wheel speed
  - k mod 10 ≠ 0: wheel speed only

## Quick Start
```bash
git clone https://github.com/Hiroshi-Okajima/multirate-kalman-filter.git
cd multirate-kalman-filter
```

Open MATLAB and run:
```matlab
MultirateKF_LMI_Rcyc_success_02  % Basic optimal design
MultirateKF_02_eig              % With eigenvalue constraints
MultirateKF_03_l2               % With l2-induced norm constraints
```

## Example Results

Automotive navigation (GPS 1Hz + Wheel speed 10Hz):
- Position RMSE: **0.600 m**
- Velocity RMSE: **0.268 m/s**
- Stable: max|λ| = **0.967**
