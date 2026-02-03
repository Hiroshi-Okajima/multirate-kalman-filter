"""
Multirate Kalman Filter Design via LMI Optimization (Python)

Author: Hiroshi Okajima
Date: February 2025

Description:
    This code implements a multirate Kalman filter for automotive navigation
    systems using LMI-based optimization through cyclic reformulation.
    The system fuses GPS measurements (1 Hz) with wheel speed sensor data (10 Hz)
    to estimate vehicle position, velocity, and acceleration.

    Key Features:
    - Handles semidefinite measurement noise covariance R_cyc naturally via LMI
    - Periodic time-varying Kalman gains computed offline
    - Dual LQR formulation ensures numerical stability
    - Trace minimization for optimal estimation error covariance

System Configuration:
    - State: [position; velocity; acceleration]  (n=3)
    - Measurements: GPS position (10% availability) + wheel speed (100%)  (m=2)
    - Period: N = 10, Sampling time: dt = 0.1s

LMI Formulation (Paper Equation (39)):
    [X,              X*A + Y*C,   X*Q^{1/2},   Y*R^{1/2}    ]
    [(X*A + Y*C)',       X,          0,           0         ]
    [(Q^{1/2})'*X,       0,          I,           0         ]
    [(R^{1/2})'*Y',      0,          0,           I         ] >= 0

Reference:
    H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter 
    Design," IEEE Access, 2025.

Required packages:
    pip install numpy scipy matplotlib cvxpy
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, solve_discrete_are
from typing import List, Tuple

# ============================================================================
# 1. System Definition
# ============================================================================

def create_system_parameters():
    """
    Define automotive navigation system parameters
    
    State: [position; velocity; acceleration]
    Output: [GPS position; wheel speed (velocity)]
    """
    n = 3      # State dimension
    p = 1      # Input dimension  
    m = 2      # Output dimension
    N = 10     # Period (GPS: 1Hz, Wheel speed: 10Hz)
    dt = 0.1   # Sampling time [s]
    
    # State-space model: x = [position; velocity; acceleration]
    A = np.array([
        [1,  dt,  0.5*dt**2],
        [0,  1,   dt],
        [0,  0,   0.8]
    ])
    
    B = np.array([[0], [0], [1]])
    
    # Output: position and velocity
    C = np.array([
        [1, 0, 0],   # GPS position
        [0, 1, 0]    # Wheel speed sensor
    ])
    
    # Noise covariances
    Q = np.diag([0.01, 0.1, 0.5])    # Process noise
    R = np.diag([1.0, 0.1])          # GPS ±1m, wheel speed ±0.1m/s
    
    return n, p, m, N, dt, A, B, C, Q, R


def create_measurement_pattern(N: int, m: int) -> List[np.ndarray]:
    """
    Create measurement pattern S_k for automotive navigation
    
    GPS (position): available every 10 steps (k mod 10 = 0)
    Wheel speed (velocity): available every step
    """
    S = []
    
    # k mod 10 = 0: GPS + wheel speed
    S.append(np.diag([1, 1]))
    
    # k mod 10 = 1-9: wheel speed only
    for i in range(1, N):
        S.append(np.diag([0, 1]))
    
    return S


# ============================================================================
# 2. Cyclic Reformulation Construction
# ============================================================================

def construct_cyclic_system(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                            Q: np.ndarray, R: np.ndarray, 
                            S: List[np.ndarray], N: int) -> Tuple:
    """
    Construct cyclic reformulation of the multirate system
    
    Following Paper Section III (Equations 24-28)
    """
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]
    
    n_cyc = N * n  # 30
    m_cyc = N * m  # 20
    p_cyc = N * p  # 10
    
    # A_cyc: Nn × Nn cyclic matrix (Equation 24)
    # [0  0  ... 0  A]
    # [A  0  ... 0  0]
    # [0  A  ... 0  0]
    # ...
    # [0  0  ... A  0]
    A_cyc = np.zeros((n_cyc, n_cyc))
    A_cyc[0:n, (N-1)*n:N*n] = A  # First block row: (1,N) position
    for i in range(1, N):
        A_cyc[i*n:(i+1)*n, (i-1)*n:i*n] = A  # Subdiagonal
    
    # B_cyc: Nn × Np cyclic matrix
    B_cyc = np.zeros((n_cyc, p_cyc))
    B_cyc[0:n, (N-1)*p:N*p] = B
    for i in range(1, N):
        B_cyc[i*n:(i+1)*n, (i-1)*p:i*p] = B
    
    # C_cyc: Nm × Nn block-diagonal matrix (Equation 25)
    # C_cyc = diag(S_0*C, S_1*C, ..., S_{N-1}*C)
    C_cyc = np.zeros((m_cyc, n_cyc))
    for i in range(N):
        C_cyc[i*m:(i+1)*m, i*n:(i+1)*n] = S[i] @ C
    
    # Q_cyc: Nn × Nn block diagonal (Equation 26)
    Q_cyc = np.kron(np.eye(N), Q)
    
    # R_cyc: Nm × Nm block diagonal (Equation 27)
    # Note: When S_k = 0, the corresponding block becomes 0 (semidefinite!)
    R_cyc = np.zeros((m_cyc, m_cyc))
    for i in range(N):
        R_cyc[i*m:(i+1)*m, i*m:(i+1)*m] = S[i] @ R @ S[i].T
    
    return A_cyc, B_cyc, C_cyc, Q_cyc, R_cyc


def construct_cyclic_sqrt_matrices(Q: np.ndarray, R: np.ndarray, 
                                    S: List[np.ndarray], N: int) -> Tuple:
    """
    Construct Q_cyc^{1/2} and R_cyc^{1/2} for LMI formulation
    
    Q_cyc^{1/2}: Cyclic structure (same pattern as A_cyc)
    R_cyc^{1/2}: Block-diagonal (reflecting measurement pattern)
    """
    n = Q.shape[0]
    m = R.shape[0]
    n_cyc = N * n
    m_cyc = N * m
    
    # Q^{1/2}: Matrix square root
    Q_sqrt = sqrtm(Q).real
    
    # Q_cyc^{1/2}: Cyclic structure
    Q_cyc_sqrt = np.zeros((n_cyc, n_cyc))
    Q_cyc_sqrt[0:n, (N-1)*n:N*n] = Q_sqrt
    for i in range(1, N):
        Q_cyc_sqrt[i*n:(i+1)*n, (i-1)*n:i*n] = Q_sqrt
    
    # R^{1/2}: Matrix square root
    R_sqrt = sqrtm(R).real
    
    # R_cyc^{1/2}: Block-diagonal (reflecting measurement pattern)
    R_cyc_sqrt = np.zeros((m_cyc, m_cyc))
    for i in range(N):
        R_cyc_sqrt[i*m:(i+1)*m, i*m:(i+1)*m] = S[i] @ R_sqrt
    
    return Q_cyc_sqrt, R_cyc_sqrt


def check_observability(A_cyc: np.ndarray, C_cyc: np.ndarray) -> Tuple[int, bool]:
    """
    Check observability of the cyclic system (Proposition 2)
    """
    n_cyc = A_cyc.shape[0]
    
    # Build observability matrix
    O = C_cyc.copy()
    Ak = np.eye(n_cyc)
    for i in range(1, n_cyc):
        Ak = Ak @ A_cyc
        O = np.vstack([O, C_cyc @ Ak])
    
    rank_O = np.linalg.matrix_rank(O)
    is_observable = (rank_O == n_cyc)
    
    return rank_O, is_observable


# ============================================================================
# 3. LMI-based Filter Design
# ============================================================================

def solve_lmi_kalman_filter(A_cyc: np.ndarray, C_cyc: np.ndarray,
                            Q_cyc_sqrt: np.ndarray, R_cyc_sqrt: np.ndarray,
                            epsilon: float = 1e-6, 
                            verbose: bool = False) -> Tuple:
    """
    Solve LMI optimization for multirate Kalman filter design
    
    Paper Equation (39):
    [X,              X*A + Y*C,      X*Q^{1/2},      Y*R^{1/2}    ]
    [(X*A + Y*C)',   X,              0,              0            ]
    [(Q^{1/2})'*X,   0,              I_{Nn},         0            ]
    [(R^{1/2})'*Y',  0,              0,              I_{Nq}       ] >= 0
    
    Decision variables:
        X ∈ R^{Nn×Nn}: symmetric positive definite (Lyapunov matrix)
        Y ∈ R^{Nn×Nq}: rectangular, Y = -X*L where L is the Kalman gain
    
    Returns:
        X_opt, Y_opt, W_opt, L_cyc, P_ss, optimal_trace
    """
    n_cyc = A_cyc.shape[0]
    m_cyc = C_cyc.shape[0]
    
    # Decision variables (matching paper notation)
    X = cp.Variable((n_cyc, n_cyc), symmetric=True)  # X = P^{-1}
    Y = cp.Variable((n_cyc, m_cyc))                   # Y = -X*L
    W = cp.Variable((n_cyc, n_cyc), symmetric=True)  # For trace minimization
    
    constraints = []
    
    # LMI #1: DARI (Equation 39)
    # Using transposed sqrt matrices to match paper notation
    # (1,2): X*A + Y*C
    # (3,1): (Q^{1/2})' * X
    # (4,1): (R^{1/2})' * Y'
    
    block_12 = X @ A_cyc + Y @ C_cyc           # X*A + Y*C
    block_31 = Q_cyc_sqrt.T @ X               # (Q^{1/2})' * X
    block_41 = R_cyc_sqrt.T @ Y.T             # (R^{1/2})' * Y'
    
    LMI1 = cp.bmat([
        [X,         block_12,   block_31.T, block_41.T],
        [block_12.T, X,         np.zeros((n_cyc, n_cyc)), np.zeros((n_cyc, m_cyc))],
        [block_31,  np.zeros((n_cyc, n_cyc)), np.eye(n_cyc), np.zeros((n_cyc, m_cyc))],
        [block_41,  np.zeros((m_cyc, n_cyc)), np.zeros((m_cyc, n_cyc)), np.eye(m_cyc)]
    ])
    constraints.append(LMI1 >> 0)
    
    # LMI #2: X > epsilon*I (positive definiteness)
    constraints.append(X >> epsilon * np.eye(n_cyc))
    
    # LMI #3: [W, I; I, X] >= 0 => W >= X^{-1} = P
    LMI3 = cp.bmat([
        [W, np.eye(n_cyc)],
        [np.eye(n_cyc), X]
    ])
    constraints.append(LMI3 >> 0)
    
    # Objective: minimize trace(W)
    objective = cp.Minimize(cp.trace(W))
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.SCS, verbose=verbose, max_iters=10000, eps=1e-9)
    except:
        try:
            problem.solve(solver=cp.CLARABEL, verbose=verbose)
        except:
            problem.solve(solver=cp.CVXOPT, verbose=verbose)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"LMI optimization failed! Status: {problem.status}")
    
    # Extract optimal values
    X_opt = X.value
    Y_opt = Y.value
    W_opt = W.value
    
    # Error covariance: P = X^{-1}
    P_ss = np.linalg.inv(X_opt)
    
    # Kalman gain recovery: L = -X^{-1}*Y (Equation 43)
    L_cyc = -np.linalg.solve(X_opt, Y_opt)
    
    return X_opt, Y_opt, W_opt, L_cyc, P_ss, problem.value


# ============================================================================
# 4. Extract Periodic Kalman Gains
# ============================================================================

def extract_periodic_gains(L_cyc: np.ndarray, N: int, n: int, m: int) -> List[np.ndarray]:
    """
    Extract periodic time-varying gains from L_cyc
    
    Following Paper Equation (44):
        L_0 is at block position (2,1) of L_cyc
        L_k is at block position (k+2 mod N, k+1) for k = 0,...,N-1
        L_{N-1} is at block position (1, N)
    
    Note: In Python (0-indexed), this becomes:
        L_k at (k+1)*n:(k+2)*n, k*m:(k+1)*m for k = 0,...,N-2
        L_{N-1} at 0:n, (N-1)*m:N*m
    """
    L = []
    
    # L_0 to L_{N-2}
    for k in range(N-1):
        L_k = L_cyc[(k+1)*n:(k+2)*n, k*m:(k+1)*m]
        L.append(L_k)
    
    # L_{N-1}
    L_Nm1 = L_cyc[0:n, (N-1)*m:N*m]
    L.append(L_Nm1)
    
    return L


# ============================================================================
# 5. Simulation
# ============================================================================

def simulate_multirate_kalman_filter(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                                      Q: np.ndarray, R: np.ndarray,
                                      S: List[np.ndarray], L: List[np.ndarray],
                                      T: int, x0: np.ndarray = None,
                                      seed: int = 42) -> Tuple:
    """
    Simulate the multirate Kalman filter
    """
    np.random.seed(seed)
    
    N = len(S)
    n = A.shape[0]
    m = C.shape[0]
    p = B.shape[1]
    
    if x0 is None:
        x0 = np.array([0, 5, 0])  # Initial: position 0m, velocity 5m/s
    
    # True state
    x_true = np.zeros((n, T))
    x_true[:, 0] = x0
    
    # Observations
    z_obs = np.zeros((m, T))
    
    # Control input (acceleration command)
    u = 0.5 * np.sin(0.05 * np.arange(T))
    
    # Generate true trajectory
    Q_sqrt = sqrtm(Q).real
    R_sqrt = sqrtm(R).real
    
    for k in range(T - 1):
        w = Q_sqrt @ np.random.randn(n)
        x_true[:, k+1] = A @ x_true[:, k] + B.flatten() * u[k] + w
    
    # Generate observations
    for k in range(T):
        v = R_sqrt @ np.random.randn(m)
        z_obs[:, k] = C @ x_true[:, k] + v
    
    # Kalman filter estimation
    x_hat = np.zeros((n, T))
    x_hat[:, 0] = x_true[:, 0]  # Perfect initial condition
    
    for k in range(T - 1):
        # Prediction
        x_pred = A @ x_hat[:, k] + B.flatten() * u[k]
        z_pred = C @ x_pred
        
        # Update with periodic gain
        idx = k % N
        innovation = z_obs[:, k+1] - z_pred
        
        x_hat[:, k+1] = x_pred + L[idx] @ innovation
    
    return x_true, x_hat, z_obs, u


# ============================================================================
# 6. Visualization
# ============================================================================

def plot_results(x_true: np.ndarray, x_hat: np.ndarray, z_obs: np.ndarray,
                 L: List[np.ndarray], eig_cl: np.ndarray, P_ss: np.ndarray,
                 N: int, n: int, m: int, dt: float):
    """
    Plot estimation results and analysis
    """
    T = x_true.shape[1]
    
    fig = plt.figure(figsize=(16, 12))
    
    # Position
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(range(T), x_true[0, :], 'k-', lw=2, label='True')
    ax1.plot(range(T), x_hat[0, :], 'b--', lw=1.5, label='Estimated')
    obs_idx = [k for k in range(T) if k % N == 0]
    ax1.plot(obs_idx, z_obs[0, obs_idx], 'go', ms=6, label='GPS')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Position (GPS: every 10 steps)')
    ax1.legend()
    ax1.grid(True)
    
    # Velocity
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(range(T), x_true[1, :], 'k-', lw=2, label='True')
    ax2.plot(range(T), x_hat[1, :], 'b--', lw=1.5, label='Estimated')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_title('Velocity (Wheel Speed: every step)')
    ax2.legend()
    ax2.grid(True)
    
    # Acceleration
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(range(T), x_true[2, :], 'k-', lw=2, label='True')
    ax3.plot(range(T), x_hat[2, :], 'b--', lw=1.5, label='Estimated')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Acceleration [m/s²]')
    ax3.set_title('Acceleration (not observed)')
    ax3.legend()
    ax3.grid(True)
    
    # Estimation errors
    error = x_true - x_hat
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(range(T), error[0, :], lw=1.5, label='Position')
    ax4.plot(range(T), error[1, :], lw=1.5, label='Velocity')
    ax4.plot(range(T), error[2, :], lw=1.5, label='Acceleration')
    ax4.axhline(0, color='k', ls='--')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Error')
    ax4.set_title('Estimation Errors')
    ax4.legend()
    ax4.grid(True)
    
    # Position error (zoomed)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(range(T), error[0, :], lw=1.5)
    ax5.axhline(0, color='k', ls='--')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Position Error [m]')
    ax5.set_title(f'Position Error (RMSE={np.sqrt(np.mean(error[0,:]**2)):.4f} m)')
    ax5.grid(True)
    
    # Velocity error (zoomed)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(range(T), error[1, :], lw=1.5)
    ax6.axhline(0, color='k', ls='--')
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Velocity Error [m/s]')
    ax6.set_title(f'Velocity Error (RMSE={np.sqrt(np.mean(error[1,:]**2)):.4f} m/s)')
    ax6.grid(True)
    
    # Closed-loop eigenvalues
    ax7 = fig.add_subplot(3, 3, 7)
    theta = np.linspace(0, 2*np.pi, 100)
    ax7.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)
    ax7.plot(np.real(eig_cl), np.imag(eig_cl), 'bo', ms=8)
    ax7.set_aspect('equal')
    ax7.set_xlabel('Real')
    ax7.set_ylabel('Imaginary')
    ax7.set_title(f'Closed-loop Eigenvalues (max|λ|={np.max(np.abs(eig_cl)):.4f})')
    ax7.grid(True)
    
    # Eigenvalue magnitudes
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.bar(range(len(eig_cl)), np.abs(eig_cl))
    ax8.axhline(1.0, color='k', ls='--', lw=2)
    ax8.set_xlabel('Index')
    ax8.set_ylabel('|λ|')
    ax8.set_title('Eigenvalue Magnitudes')
    ax8.set_ylim([0, 1.1])
    ax8.grid(True)
    
    # Periodic gain norms
    ax9 = fig.add_subplot(3, 3, 9)
    L_norms = [np.linalg.norm(L[k], 'fro') for k in range(N)]
    ax9.bar(range(N), L_norms)
    ax9.set_xlabel('k mod N')
    ax9.set_ylabel('||L_k|| (Frobenius)')
    ax9.set_title('Periodic Kalman Gain Norms')
    ax9.grid(True)
    
    plt.suptitle('Multirate Kalman Filter: Automotive Navigation (n=3, m=2, N=10)', 
                 fontsize=14)
    plt.tight_layout()
    
    return fig


def plot_gain_analysis(L: List[np.ndarray], P_ss: np.ndarray, R_cyc: np.ndarray,
                       N: int, n: int, m: int):
    """
    Plot detailed gain analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gain norm
    ax1 = axes[0, 0]
    L_norms = [np.linalg.norm(L[k], 'fro') for k in range(N)]
    ax1.bar(range(N), L_norms)
    ax1.set_xlabel('k mod 10')
    ax1.set_ylabel('||L_k|| (Frobenius)')
    ax1.set_title('Kalman Gain Norm (Periodic)')
    ax1.grid(True)
    
    # Gain contribution to each state
    ax2 = axes[0, 1]
    L_contributions = np.zeros((N, n))
    for k in range(N):
        for j in range(n):
            L_contributions[k, j] = np.linalg.norm(L[k][j, :])
    
    x = np.arange(N)
    width = 0.25
    ax2.bar(x - width, L_contributions[:, 0], width, label='Position')
    ax2.bar(x, L_contributions[:, 1], width, label='Velocity')
    ax2.bar(x + width, L_contributions[:, 2], width, label='Acceleration')
    ax2.set_xlabel('k mod 10')
    ax2.set_ylabel('Gain Magnitude')
    ax2.set_title('Kalman Gain Contribution to Each State')
    ax2.legend()
    ax2.grid(True)
    
    # P matrix eigenvalues
    ax3 = axes[1, 0]
    P_eigs = np.sort(np.linalg.eigvals(P_ss).real)[::-1]
    ax3.semilogy(range(1, len(P_eigs)+1), P_eigs, 'bo-', lw=1.5)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('P Matrix Eigenvalues')
    ax3.grid(True)
    
    # R_cyc sparsity pattern
    ax4 = axes[1, 1]
    ax4.spy(R_cyc, markersize=3)
    ax4.set_title('R_cyc Sparsity Pattern')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    
    plt.suptitle('Gain Analysis: Automotive Navigation', fontsize=14)
    plt.tight_layout()
    
    return fig


# ============================================================================
# Main Program
# ============================================================================

def main():
    print("=" * 60)
    print("Multirate Kalman Filter (LMI Design)")
    print("Automotive Navigation System")
    print("=" * 60)
    
    # 1. System Definition
    n, p, m, N, dt, A, B, C, Q, R = create_system_parameters()
    S = create_measurement_pattern(N, m)
    
    print(f"\nSystem:")
    print(f"  State dimension: n={n} [position, velocity, acceleration]")
    print(f"  Output dimension: m={m} [GPS position, wheel speed]")
    print(f"  Observation period: N={N}")
    print(f"  Sampling time: dt={dt:.2f} [s]")
    print(f"  Eigenvalues of A: {np.linalg.eigvals(A)}")
    
    print(f"\nSensor Specifications:")
    print(f"  GPS (position): 1Hz (every {N} steps), accuracy ±{np.sqrt(R[0,0]):.1f}m")
    print(f"  Wheel speed (velocity): 10Hz (every step), accuracy ±{np.sqrt(R[1,1]):.1f}m/s")
    
    print(f"\nMeasurement Pattern:")
    print(f"  k mod {N} = 0: GPS ON + Wheel speed ON")
    print(f"  k mod {N} = 1-{N-1}: GPS OFF + Wheel speed ON")
    
    # 2. Cyclic Reformulation
    print("\n" + "=" * 40)
    print("Cyclic Reformulation")
    print("=" * 40)
    
    A_cyc, B_cyc, C_cyc, Q_cyc, R_cyc = construct_cyclic_system(A, B, C, Q, R, S, N)
    Q_cyc_sqrt, R_cyc_sqrt = construct_cyclic_sqrt_matrices(Q, R, S, N)
    
    n_cyc = N * n
    m_cyc = N * m
    
    print(f"\nCyclic System Dimensions:")
    print(f"  A_cyc: {A_cyc.shape}")
    print(f"  B_cyc: {B_cyc.shape}")
    print(f"  C_cyc: {C_cyc.shape}")
    print(f"  Q_cyc: {Q_cyc.shape}")
    print(f"  R_cyc: {R_cyc.shape} (rank: {np.linalg.matrix_rank(R_cyc)})")
    
    print(f"\nR_cyc Structure:")
    print(f"  Number of nonzero diagonal elements: {np.sum(np.diag(R_cyc) != 0)}")
    print(f"  R_cyc rank: {np.linalg.matrix_rank(R_cyc)} / {m_cyc} (SEMIDEFINITE)")
    
    # 3. Observability Check
    rank_O, is_obs = check_observability(A_cyc, C_cyc)
    print(f"\nObservability:")
    print(f"  Rank: {rank_O} / {n_cyc}")
    print(f"  Result: {'Observable' if is_obs else 'NOT Observable'}")
    
    # 4. LMI Optimization
    print("\n" + "=" * 40)
    print("LMI Optimization")
    print("=" * 40)
    
    print(f"\nDecision variables:")
    print(f"  X: {n_cyc}×{n_cyc} symmetric positive definite")
    print(f"  Y: {n_cyc}×{m_cyc} rectangular (Y = -X*L)")
    print(f"  W: {n_cyc}×{n_cyc} symmetric (covariance bound)")
    
    print("\nSolving LMI...")
    X_opt, Y_opt, W_opt, L_cyc, P_ss, opt_trace = solve_lmi_kalman_filter(
        A_cyc, C_cyc, Q_cyc_sqrt, R_cyc_sqrt, verbose=False
    )
    
    print(f"\nLMI Results:")
    print(f"  Optimal trace(W) = {opt_trace:.6f} (upper bound on trace(P))")
    print(f"  Actual trace(P) = {np.trace(P_ss):.6f}")
    print(f"  L_cyc size: {L_cyc.shape}")
    
    # Closed-loop stability
    A_cl = A_cyc - L_cyc @ C_cyc
    eig_cl = np.linalg.eigvals(A_cl)
    max_eig = np.max(np.abs(eig_cl))
    
    print(f"\nStability Analysis:")
    print(f"  Max |eigenvalue| of A_cyc - L_cyc*C_cyc: {max_eig:.10f}")
    print(f"  Result: {'STABLE' if max_eig < 1 else 'UNSTABLE'}")
    
    # 5. Extract Periodic Gains
    print("\n" + "=" * 40)
    print("Periodic Kalman Gains")
    print("=" * 40)
    
    L = extract_periodic_gains(L_cyc, N, n, m)
    
    print(f"\nL_0 (k mod {N} = 0, GPS + wheel speed):")
    print(L[0])
    
    print(f"\nL_1 (k mod {N} = 1, wheel speed only):")
    print(L[1])
    
    print(f"\nL_5 (k mod {N} = 5, wheel speed only):")
    print(L[5])
    
    # 6. Simulation
    print("\n" + "=" * 40)
    print("Simulation")
    print("=" * 40)
    
    T = 200
    x0 = np.array([0, 5, 0])  # Initial: position 0m, velocity 5m/s
    
    x_true, x_hat, z_obs, u = simulate_multirate_kalman_filter(
        A, B, C, Q, R, S, L, T, x0, seed=42
    )
    
    print(f"\nDriving simulation:")
    print(f"  Total time: {T*dt:.1f} seconds")
    print(f"  Total distance traveled: {x_true[0, -1]:.1f} m")
    print(f"  Maximum velocity: {np.max(x_true[1, :]):.1f} m/s ({np.max(x_true[1, :])*3.6:.1f} km/h)")
    
    # 7. Performance Evaluation
    print("\n" + "=" * 40)
    print("Performance Evaluation")
    print("=" * 40)
    
    error = x_true - x_hat
    rmse = np.sqrt(np.mean(error**2, axis=1))
    max_error = np.max(np.abs(error), axis=1)
    
    print(f"\nRMSE:")
    print(f"  Position:     {rmse[0]:.4f} [m]")
    print(f"  Velocity:     {rmse[1]:.4f} [m/s]")
    print(f"  Acceleration: {rmse[2]:.4f} [m/s²]")
    
    print(f"\nMaximum Error:")
    print(f"  Position:     {max_error[0]:.4f} [m]")
    print(f"  Velocity:     {max_error[1]:.4f} [m/s]")
    print(f"  Acceleration: {max_error[2]:.4f} [m/s²]")
    
    # 8. Plot Results
    print("\n" + "=" * 40)
    print("Generating plots...")
    print("=" * 40)
    
    fig1 = plot_results(x_true, x_hat, z_obs, L, eig_cl, P_ss, N, n, m, dt)
    fig1.savefig('/home/claude/multirate_kf_results.png', dpi=150, bbox_inches='tight')
    
    fig2 = plot_gain_analysis(L, P_ss, R_cyc, N, n, m)
    fig2.savefig('/home/claude/multirate_kf_gains.png', dpi=150, bbox_inches='tight')
    
    print("\nPlots saved to:")
    print("  multirate_kf_results.png")
    print("  multirate_kf_gains.png")
    
    print("\n" + "=" * 60)
    print("Program completed successfully")
    print("=" * 60)
    
    return L, P_ss, eig_cl, rmse


if __name__ == "__main__":
    L, P_ss, eig_cl, rmse = main()
