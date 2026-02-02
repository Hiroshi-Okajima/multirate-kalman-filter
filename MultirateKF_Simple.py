"""
Simplified Multirate Kalman Filter Design via LMI Optimization (Python)

Author: Hiroshi Okajima
Date: February 2025

Description:
    Simplified version with state dimension n=1, output m=1, period N=6
    This allows easy modification to periods N=1, 2, 3, 6 by adjusting S_k

System:
    x(k+1) = A*x(k) + B*u(k) + w(k),  w ~ N(0, Q)
    y(k) = S_k*C*x(k) + S_k*v(k),     v ~ N(0, R)

Reference:
    H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter 
    Design," IEEE Access, 2025.

Required packages:
    pip install numpy scipy matplotlib cvxpy
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from typing import List, Tuple

# ============================================================================
# 1. System Definition
# ============================================================================

def create_system_parameters():
    """Define system parameters"""
    n = 1      # State dimension
    p = 1      # Input dimension  
    m = 1      # Output dimension
    N = 6      # Period (can test N=1,2,3,6 by changing S pattern)
    dt = 0.1   # Sampling time [s]
    
    # First-order system: position tracking
    A = np.array([[0.95]])   # Stable system (eigenvalue < 1)
    B = np.array([[0.1]])    # Input gain
    C = np.array([[1.0]])    # Full state observation when available
    
    # Noise covariances
    Q = np.array([[0.1]])    # Process noise variance
    R = np.array([[1.0]])    # Measurement noise variance
    
    return n, p, m, N, dt, A, B, C, Q, R


def create_measurement_pattern(N: int, pattern_type: int = 1) -> List[np.ndarray]:
    """
    Create measurement pattern S_k
    
    pattern_type:
        1: Sensor available every 6 steps (N=6 effective)
        2: Sensor available every 3 steps (N=3 effective within N=6)
        3: Sensor available every 2 steps (N=2 effective within N=6)
        4: Sensor available every step (standard Kalman filter, N=1 effective)
    """
    S = []
    
    if pattern_type == 1:
        # Pattern 1: Sensor available every 6 steps
        S = [np.array([[1]]),   # k mod 6 = 0: measurement available
             np.array([[0]]),   # k mod 6 = 1: no measurement
             np.array([[0]]),   # k mod 6 = 2: no measurement
             np.array([[0]]),   # k mod 6 = 3: no measurement
             np.array([[0]]),   # k mod 6 = 4: no measurement
             np.array([[0]])]   # k mod 6 = 5: no measurement
             
    elif pattern_type == 2:
        # Pattern 2: Sensor available every 3 steps
        S = [np.array([[1]]),   # k mod 6 = 0: measurement
             np.array([[0]]),   # k mod 6 = 1: no measurement
             np.array([[0]]),   # k mod 6 = 2: no measurement
             np.array([[1]]),   # k mod 6 = 3: measurement
             np.array([[0]]),   # k mod 6 = 4: no measurement
             np.array([[0]])]   # k mod 6 = 5: no measurement
             
    elif pattern_type == 3:
        # Pattern 3: Sensor available every 2 steps
        S = [np.array([[1]]),   # k mod 6 = 0: measurement
             np.array([[0]]),   # k mod 6 = 1: no measurement
             np.array([[1]]),   # k mod 6 = 2: measurement
             np.array([[0]]),   # k mod 6 = 3: no measurement
             np.array([[1]]),   # k mod 6 = 4: measurement
             np.array([[0]])]   # k mod 6 = 5: no measurement
             
    elif pattern_type == 4:
        # Pattern 4: Sensor available every step (standard Kalman filter)
        S = [np.array([[1]]) for _ in range(N)]
        
    return S


# ============================================================================
# 2. Cyclic Reformulation Construction
# ============================================================================

def construct_cyclic_system(A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                            Q: np.ndarray, R: np.ndarray, 
                            S: List[np.ndarray], N: int) -> Tuple:
    """
    Construct cyclic reformulation of the multirate system
    
    Returns:
        A_cyc, B_cyc, C_cyc, Q_cyc, R_cyc
    """
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]
    
    # A_cyc: Nn × Nn cyclic matrix
    # Structure: [0 0 ... 0 A; A 0 ... 0 0; 0 A ... 0 0; ...; 0 0 ... A 0]
    A_cyc = np.zeros((N*n, N*n))
    A_cyc[0:n, (N-1)*n:N*n] = A  # First block row: (1,N) position
    for i in range(1, N):
        A_cyc[i*n:(i+1)*n, (i-1)*n:i*n] = A  # Subdiagonal
    
    # B_cyc: Nn × Np cyclic matrix (same structure as A_cyc)
    B_cyc = np.zeros((N*n, N*p))
    B_cyc[0:n, (N-1)*p:N*p] = B
    for i in range(1, N):
        B_cyc[i*n:(i+1)*n, (i-1)*p:i*p] = B
    
    # C_cyc: Nm × Nn block-diagonal matrix
    # C_cyc = diag(S_0*C, S_1*C, ..., S_{N-1}*C)
    C_cyc = np.zeros((N*m, N*n))
    for i in range(N):
        C_cyc[i*m:(i+1)*m, i*n:(i+1)*n] = S[i] @ C
    
    # Q_cyc: Nn × Nn (block diagonal with Q)
    Q_cyc = np.kron(np.eye(N), Q)
    
    # R_cyc: Nm × Nm (block diagonal with S_k*R*S_k')
    # Note: When S_k = 0, the corresponding block is 0 (semidefinite!)
    R_cyc = np.zeros((N*m, N*m))
    for i in range(N):
        R_cyc[i*m:(i+1)*m, i*m:(i+1)*m] = S[i] @ R @ S[i].T
    
    return A_cyc, B_cyc, C_cyc, Q_cyc, R_cyc


def construct_cyclic_sqrt_matrices(Q: np.ndarray, R: np.ndarray, 
                                   S: List[np.ndarray], N: int,
                                   epsilon: float = 1e-8) -> Tuple:
    """
    Construct Q_cyc_sqrt and R_cyc_sqrt for LMI formulation
    
    Q_cyc_sqrt has cyclic structure (same pattern as A_cyc)
    R_cyc_sqrt is block diagonal with regularization
    """
    n = Q.shape[0]
    m = R.shape[0]
    
    # Q_sqrt
    Q_sqrt = np.linalg.cholesky(Q)
    
    # Q_cyc_sqrt: Cyclic structure (same pattern as A_cyc)
    Q_cyc_sqrt = np.zeros((N*n, N*n))
    Q_cyc_sqrt[0:n, (N-1)*n:N*n] = Q_sqrt
    for i in range(1, N):
        Q_cyc_sqrt[i*n:(i+1)*n, (i-1)*n:i*n] = Q_sqrt
    
    # R_cyc_sqrt: Block diagonal (with regularization for semidefinite case)
    R_cyc_sqrt = np.zeros((N*m, N*m))
    for i in range(N):
        R_block = S[i] @ R @ S[i].T + epsilon * np.eye(m)
        R_cyc_sqrt[i*m:(i+1)*m, i*m:(i+1)*m] = np.linalg.cholesky(R_block)
    
    return Q_cyc_sqrt, R_cyc_sqrt


# ============================================================================
# 3. Observability Check
# ============================================================================

def check_observability(A_cyc: np.ndarray, C_cyc: np.ndarray) -> Tuple[int, bool]:
    """Check observability of the cyclic system"""
    n_cyc = A_cyc.shape[0]
    
    # Construct observability matrix
    O = C_cyc.copy()
    Ak = np.eye(n_cyc)
    for i in range(1, n_cyc):
        Ak = Ak @ A_cyc
        O = np.vstack([O, C_cyc @ Ak])
    
    rank_O = np.linalg.matrix_rank(O)
    is_observable = (rank_O == n_cyc)
    
    return rank_O, is_observable


# ============================================================================
# 4. LMI-based Filter Design (Dual LQR Formulation)
# ============================================================================

def solve_lmi_kalman_filter(A_cyc: np.ndarray, C_cyc: np.ndarray,
                            Q_cyc_sqrt: np.ndarray, R_cyc_sqrt: np.ndarray,
                            epsilon: float = 1e-6, 
                            verbose: bool = True) -> Tuple:
    """
    Solve LMI optimization for multirate Kalman filter design
    
    Uses dual LQR formulation with cvxpy following paper equations (39)-(43)
    
    In MATLAB code:
    - Y is m_cyc × n_cyc (Y = K*X where K is dual gain)
    - Block (4,1) is R^{1/2}*Y
    
    Returns:
        X_opt, Y_opt, W_opt, K_ss, P_ss
    """
    n_cyc = A_cyc.shape[0]
    m_cyc = C_cyc.shape[0]
    
    # Dual system matrices (same as MATLAB)
    Ad = A_cyc.T   # Dual A = A_cyc'
    Bd = C_cyc.T   # Dual B = C_cyc'
    
    # Decision variables (matching MATLAB dimensions)
    # X: n_cyc × n_cyc symmetric (Lyapunov matrix)
    # Y: m_cyc × n_cyc (Y = K*X where K is dual LQR gain)
    X = cp.Variable((n_cyc, n_cyc), symmetric=True)
    Y = cp.Variable((m_cyc, n_cyc))
    W = cp.Variable((n_cyc, n_cyc), symmetric=True)
    
    constraints = []
    
    # LMI #1: Stability and Performance (Equation 39 in paper)
    # Paper defines Y ∈ R^{Nn×Nq}, but MATLAB/Python uses Y ∈ R^{Nq×Nn} (transposed)
    # 
    # Paper equation (39):
    # [X,                (XA + YC)^T,      X*Q^{1/2},         Y*R^{1/2}       ]
    # [XA + YC,          X,                0,                 0               ]
    # [(Q^{1/2})^T*X,    0,                I,                 0               ]
    # [(R^{1/2})^T*Y^T,  0,                0,                 I               ]
    #
    # With Y_matlab = Y_paper^T:
    # (R^{1/2})^T * Y_paper^T = (R^{1/2})^T * Y_matlab
    
    block_11 = X
    block_21 = Ad @ X + Bd @ Y           # A_cyc' * X + C_cyc' * Y
    block_31 = Q_cyc_sqrt.T @ X          # (Q^{1/2})^T * X  (paper eq.39)
    block_41 = R_cyc_sqrt.T @ Y          # (R^{1/2})^T * Y_matlab = (R^{1/2})^T * Y_paper^T
    
    block_22 = X
    block_33 = np.eye(n_cyc)
    block_44 = np.eye(m_cyc)
    
    LMI1 = cp.bmat([
        [block_11, block_21.T, block_31.T, block_41.T],
        [block_21, block_22, np.zeros((n_cyc, n_cyc)), np.zeros((n_cyc, m_cyc))],
        [block_31, np.zeros((n_cyc, n_cyc)), block_33, np.zeros((n_cyc, m_cyc))],
        [block_41, np.zeros((m_cyc, n_cyc)), np.zeros((m_cyc, n_cyc)), block_44]
    ])
    constraints.append(LMI1 >> 0)
    
    # LMI #2: X > epsilon*I (Equation 40)
    constraints.append(X >> epsilon * np.eye(n_cyc))
    
    # LMI #3: [W, I; I, X] >= 0 => W >= X^{-1} (Equation 41)
    LMI3 = cp.bmat([
        [W, np.eye(n_cyc)],
        [np.eye(n_cyc), X]
    ])
    constraints.append(LMI3 >> 0)
    
    # Objective: minimize trace(W) (Equation 42)
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
    
    # Error covariance (equation 45): P = X^{-1}
    P_ss = np.linalg.inv(X_opt)
    
    # Kalman gain: K_ss = (-P_ss @ Y_opt).T
    # This matches MATLAB implementation
    K_ss = (-P_ss @ Y_opt).T
    
    return X_opt, Y_opt, W_opt, K_ss, P_ss, problem.value


# ============================================================================
# 5. Extract Periodic Kalman Gains
# ============================================================================

def extract_periodic_gains(K_ss: np.ndarray, N: int, n: int, m: int) -> List[np.ndarray]:
    """
    Extract periodic Kalman gains from cyclic gain matrix
    
    Following paper equation (44):
    Due to cyclic structure of A_cyc (equation 24):
    - L_0 = L^{(2),(1)} (block at row 2, col 1)
    - L_k = L^{(k+2),(k+1)} for k = 1, ..., N-2
    - L_{N-1} = L^{(1),(N)}
    
    This indexing reflects the cyclic shift structure where the
    measurement at time k (in block k+1 of y_cyc) affects the
    state estimate at time k+1 (in block (k+2) mod N of x_cyc).
    """
    L = []
    
    # L_0: block (2,1) in 1-indexed = block (1,0) in 0-indexed
    L.append(K_ss[1*n:2*n, 0*m:1*m])
    
    # L_k for k = 1, ..., N-2: block (k+2, k+1) in 1-indexed
    for k in range(1, N-1):
        row_block = k + 1  # 0-indexed: k+2-1 = k+1
        col_block = k      # 0-indexed: k+1-1 = k
        L.append(K_ss[row_block*n:(row_block+1)*n, col_block*m:(col_block+1)*m])
    
    # L_{N-1}: block (1, N) in 1-indexed = block (0, N-1) in 0-indexed
    L.append(K_ss[0*n:1*n, (N-1)*m:N*m])
    
    return L


# ============================================================================
# 6. Simulation
# ============================================================================

def simulate_multirate_kalman_filter(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                                     Q: np.ndarray, R: np.ndarray,
                                     S: List[np.ndarray], L: List[np.ndarray],
                                     T: int, x0: float = 5.0,
                                     seed: int = 42) -> Tuple:
    """
    Simulate multirate Kalman filter
    
    Returns:
        x_true, x_hat, y_obs, u
    """
    np.random.seed(seed)
    N = len(S)
    n = A.shape[0]
    m = C.shape[0]
    
    # True state
    x_true = np.zeros((n, T))
    x_true[:, 0] = x0
    
    # Observations
    y_obs = np.zeros((m, T))
    
    # Control input
    u = 0.5 * np.sin(0.1 * np.arange(T))
    
    # Generate true trajectory
    Q_sqrt = np.linalg.cholesky(Q)
    R_sqrt = np.linalg.cholesky(R)
    
    for k in range(T - 1):
        w = Q_sqrt @ np.random.randn(n)
        x_true[:, k+1] = A @ x_true[:, k] + B.flatten() * u[k] + w
    
    # Generate observations
    for k in range(T):
        v = R_sqrt @ np.random.randn(m)
        y_obs[:, k] = C @ x_true[:, k] + v
    
    # Kalman filter estimation
    x_hat = np.zeros((n, T))
    x_hat[:, 0] = x_true[:, 0]  # Perfect initial condition
    
    for k in range(T - 1):
        # Prediction
        x_pred = A @ x_hat[:, k] + B.flatten() * u[k]
        y_pred = C @ x_pred
        
        # Update with periodic gain
        idx = k % N
        S_k = S[idx][0, 0]  # Scalar for 1D case
        
        if S_k == 1:
            # Measurement available
            innovation = y_obs[:, k+1] - y_pred
            x_hat[:, k+1] = x_pred + L[idx] @ innovation
        else:
            # No measurement
            x_hat[:, k+1] = x_pred
    
    return x_true, x_hat, y_obs, u


# ============================================================================
# 7. Standard Kalman Filter for Comparison
# ============================================================================

def standard_kalman_filter(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                          Q: np.ndarray, R: np.ndarray,
                          x_true: np.ndarray, y_obs: np.ndarray,
                          u: np.ndarray) -> Tuple:
    """
    Standard Kalman filter (all measurements available)
    """
    # Solve DARE
    P_std = solve_discrete_are(A.T, C.T, Q, R)
    K_std = A @ P_std @ C.T @ np.linalg.inv(C @ P_std @ C.T + R)
    
    T = x_true.shape[1]
    n = A.shape[0]
    
    x_hat_std = np.zeros((n, T))
    x_hat_std[:, 0] = x_true[:, 0]
    
    for k in range(T - 1):
        x_pred = A @ x_hat_std[:, k] + B.flatten() * u[k]
        y_pred = C @ x_pred
        innovation = y_obs[:, k+1] - y_pred
        x_hat_std[:, k+1] = x_pred + K_std @ innovation
    
    return x_hat_std, K_std, P_std


# ============================================================================
# 8. Plotting Functions
# ============================================================================

def plot_results(x_true: np.ndarray, x_hat: np.ndarray, y_obs: np.ndarray,
                 S: List[np.ndarray], eig_cl: np.ndarray, L: List[np.ndarray],
                 K_ss: np.ndarray, rmse: float, max_eig: float,
                 n: int, m: int, N: int):
    """
    Plot simulation results
    
    Note: K_ss matrix visualization (subplot 6 in MATLAB) is simplified
    as matplotlib's imagesc equivalent requires additional handling
    """
    T = x_true.shape[1]
    error = x_true - x_hat
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Plot 1: State estimation
    ax1 = axes[0, 0]
    ax1.plot(range(T), x_true.flatten(), 'k-', linewidth=2, label='True')
    ax1.plot(range(T), x_hat.flatten(), 'b--', linewidth=1.5, label='Estimated')
    # Mark observation times
    obs_times = [k for k in range(T) if S[k % N][0, 0] == 1]
    ax1.plot(obs_times, y_obs.flatten()[obs_times], 'ro', markersize=4, label='Observations')
    ax1.set_xlabel('Time step k')
    ax1.set_ylabel('State x')
    ax1.set_title('State Estimation')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # Plot 2: Estimation error
    ax2 = axes[0, 1]
    ax2.plot(range(T), error.flatten(), linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.set_xlabel('Time step k')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Estimation Error (RMSE = {rmse:.4f})')
    ax2.grid(True)
    
    # Plot 3: Measurement pattern
    ax3 = axes[0, 2]
    pattern = [S[k % N][0, 0] for k in range(T)]
    ax3.stem(range(T), pattern, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax3.set_xlabel('Time step k')
    ax3.set_ylabel('S_k')
    ax3.set_title('Measurement Availability Pattern')
    ax3.set_ylim([-0.1, 1.1])
    ax3.grid(True)
    
    # Plot 4: Closed-loop eigenvalues
    ax4 = axes[1, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, label='Unit circle')
    ax4.plot(np.real(eig_cl), np.imag(eig_cl), 'bo', markersize=10, label='Eigenvalues')
    ax4.set_aspect('equal')
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Imaginary')
    ax4.set_title(f'Closed-loop Eigenvalues (max|λ| = {max_eig:.4f})')
    ax4.legend(loc='best')
    ax4.grid(True)
    
    # Plot 5: Periodic gains
    ax5 = axes[1, 1]
    L_vals = [L[k][0, 0] for k in range(N)]
    ax5.bar(range(N), L_vals)
    ax5.set_xlabel('k mod N')
    ax5.set_ylabel('L_k')
    ax5.set_title('Periodic Kalman Gains')
    ax5.grid(True)
    
    # Plot 6: K_ss matrix visualization
    ax6 = axes[1, 2]
    im = ax6.imshow(K_ss, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax6)
    ax6.set_xlabel('Column (output block)')
    ax6.set_ylabel('Row (state block)')
    ax6.set_title('K_ss Matrix Structure')
    
    plt.suptitle(f'Multirate Kalman Filter: n={n}, m={m}, N={N}', fontsize=14)
    plt.tight_layout()
    plt.savefig('multirate_kf_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main(pattern_type: int = 1, verbose: bool = True):
    """
    Main function to run multirate Kalman filter design
    
    Args:
        pattern_type: 1-4 for different measurement patterns
        verbose: Whether to print detailed information
    """
    print("=" * 60)
    print("Simplified Multirate Kalman Filter")
    print("(LMI Design with Cyclic Reformulation - Python Version)")
    print("=" * 60)
    print()
    
    # 1. System Definition
    n, p, m, N, dt, A, B, C, Q, R = create_system_parameters()
    S = create_measurement_pattern(N, pattern_type)
    
    print("System Parameters:")
    print(f"  State dimension: n = {n}")
    print(f"  Output dimension: m = {m}")
    print(f"  Period: N = {N}")
    print(f"  System matrix A = {A[0,0]:.3f} (eigenvalue)")
    print(f"  Process noise Q = {Q[0,0]:.3f}")
    print(f"  Measurement noise R = {R[0,0]:.3f}")
    print()
    
    # Display measurement pattern
    print("Measurement Pattern S_k:")
    for i in range(N):
        status = "(measurement available)" if S[i][0,0] == 1 else "(no measurement)"
        print(f"  S_{i} = {int(S[i][0,0])} {status}")
    print()
    
    # 2. Cyclic Reformulation
    print("=" * 40)
    print("Cyclic Reformulation")
    print("=" * 40)
    
    A_cyc, B_cyc, C_cyc, Q_cyc, R_cyc = construct_cyclic_system(A, B, C, Q, R, S, N)
    
    print(f"Cyclic System Dimensions:")
    print(f"  A_cyc: {A_cyc.shape}")
    print(f"  B_cyc: {B_cyc.shape}")
    print(f"  C_cyc: {C_cyc.shape}")
    print(f"  Q_cyc: {Q_cyc.shape}")
    print(f"  R_cyc: {R_cyc.shape}")
    print()
    
    print("A_cyc matrix:")
    print(A_cyc)
    print()
    
    print("C_cyc matrix (diagonal shows S_k*C):")
    print(C_cyc)
    print()
    
    print("R_cyc diagonal (shows S_k*R*S_k'):")
    print(np.diag(R_cyc))
    print()
    
    # Check R_cyc rank (key observation from paper)
    rank_R = np.linalg.matrix_rank(R_cyc)
    print(f"R_cyc rank: {rank_R} / {N*m}", end="")
    if rank_R < N*m:
        print(" (SEMIDEFINITE - standard DARE cannot be used!)")
    else:
        print(" (positive definite - standard DARE could be used)")
    print()
    
    # 3. Observability Check
    rank_O, is_observable = check_observability(A_cyc, C_cyc)
    print("Observability Check:")
    print(f"  Observability matrix rank: {rank_O} / {N*n}")
    print(f"  Result: {'Observable' if is_observable else 'NOT Observable (filter may not work!)'}")
    print()
    
    # 4. LMI-based Filter Design
    print("=" * 40)
    print("LMI Optimization (Dual LQR)")
    print("=" * 40)
    
    epsilon = 1e-8
    Q_cyc_sqrt, R_cyc_sqrt = construct_cyclic_sqrt_matrices(Q, R, S, N, epsilon)
    
    print(f"Dual system:")
    print(f"  Ad = A_cyc': {A_cyc.T.shape}")
    print(f"  Bd = C_cyc': {C_cyc.T.shape}")
    print(f"  R_cyc regularization: epsilon = {epsilon:.1e}")
    print()
    
    # Solve LMI
    print("Solving LMI optimization...")
    X_opt, Y_opt, W_opt, K_ss, P_ss, opt_trace = solve_lmi_kalman_filter(
        A_cyc, C_cyc, Q_cyc_sqrt, R_cyc_sqrt, 
        epsilon=1e-6, verbose=False
    )
    
    print()
    print("LMI Results:")
    print(f"  Optimal trace(W) = {opt_trace:.6f} (upper bound on trace(P))")
    print(f"  Actual trace(P) = {np.trace(P_ss):.6f}")
    print(f"  K_ss size: {K_ss.shape}")
    print()
    
    # Closed-loop stability check
    A_cl = A_cyc - K_ss @ C_cyc
    eig_cl = np.linalg.eigvals(A_cl)
    max_eig = np.max(np.abs(eig_cl))
    
    print("Stability Analysis:")
    print(f"  Max |eigenvalue| of A_cyc - K_ss*C_cyc: {max_eig:.6f}")
    print(f"  Result: {'STABLE' if max_eig < 1 else 'UNSTABLE'}")
    print()
    
    # 5. Extract Periodic Kalman Gains
    print("=" * 40)
    print("Periodic Kalman Gains L_k")
    print("=" * 40)
    
    L = extract_periodic_gains(K_ss, N, n, m)
    
    print("Periodic gains:")
    for k in range(N):
        status = "(measurement available)" if S[k][0,0] == 1 else "(no measurement, gain has no effect)"
        print(f"  L_{k} = {L[k][0,0]:.6f}  {status}")
    print()
    
    # 6. Simulation
    print("=" * 40)
    print("Simulation")
    print("=" * 40)
    
    T = 100
    x_true, x_hat, y_obs, u = simulate_multirate_kalman_filter(
        A, B, C, Q, R, S, L, T, x0=5.0, seed=42
    )
    
    # Performance metrics
    error = x_true - x_hat
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    print(f"Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Max error: {max_error:.4f}")
    print()
    
    # 7. Compare with Standard Kalman Filter
    print("=" * 40)
    print("Comparison with Standard Kalman Filter")
    print("=" * 40)
    
    x_hat_std, K_std, P_std = standard_kalman_filter(A, B, C, Q, R, x_true, y_obs, u)
    
    error_std = x_true - x_hat_std
    rmse_std = np.sqrt(np.mean(error_std**2))
    
    print(f"Standard Kalman filter (all S_k = 1):")
    print(f"  Steady-state gain: K = {K_std[0,0]:.6f}")
    print(f"  Steady-state covariance: P = {P_std[0,0]:.6f}")
    print()
    
    print("Performance comparison:")
    print(f"  Multirate KF RMSE: {rmse:.4f}")
    print(f"  Standard KF RMSE (if all obs available): {rmse_std:.4f}")
    
    # Count actual observations in multirate case
    n_obs = sum([1 for k in range(T) if S[k % N][0,0] == 1])
    print(f"  Observations used: {n_obs} / {T} ({100*n_obs/T:.1f}%)")
    print()
    
    # 8. Plot results
    print("=" * 40)
    print("Generating plots...")
    print("=" * 40)
    
    plot_results(x_true, x_hat, y_obs, S, eig_cl, L, K_ss, rmse, max_eig, n, m, N)
    
    print()
    print("=" * 60)
    print("Program completed successfully")
    print("=" * 60)
    
    return {
        'K_ss': K_ss,
        'P_ss': P_ss,
        'L': L,
        'eig_cl': eig_cl,
        'rmse': rmse,
        'x_true': x_true,
        'x_hat': x_hat
    }


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    # Run with pattern_type 1 (sensor available every 6 steps)
    # Change to 2, 3, or 4 for different patterns
    results = main(pattern_type=1, verbose=True)
