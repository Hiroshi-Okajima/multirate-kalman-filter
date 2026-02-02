%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simplified Multirate Kalman Filter Design via LMI Optimization
%
% Author: Hiroshi Okajima
% Date: February 2025
%
% Description:
%   Simplified version with state dimension n=1, output m=1, period N=6
%   This allows easy modification to periods N=1, 2, 3, 6 by adjusting S_k
%
% System:
%   x(k+1) = A*x(k) + B*u(k) + w(k),  w ~ N(0, Q)
%   y(k) = S_k*C*x(k) + S_k*v(k),     v ~ N(0, R)
%
% Reference:
%   H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter 
%   Design," IEEE Access, 2025.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% 1. System Definition
n = 1;      % State dimension
p = 1;      % Input dimension  
m = 1;      % Output dimension
N = 6;      % Period (can test N=1,2,3,6 by changing S pattern)
dt = 0.1;   % Sampling time [s]

% First-order system: position tracking
A = 0.95;   % Stable system (eigenvalue < 1)
B = 0.1;    % Input gain
C = 1;      % Full state observation when available

% Noise covariances
Q = 0.1;    % Process noise variance
R = 1.0;    % Measurement noise variance

%% Measurement Pattern Selection
% Choose one of the following patterns by uncommenting:

% Pattern 1: Sensor available every 6 steps (N=6 effective)
S = cell(N, 1);
S{1} = 1;   % k mod 6 = 0: measurement available
S{2} = 0;   % k mod 6 = 1: no measurement
S{3} = 0;   % k mod 6 = 2: no measurement
S{4} = 0;   % k mod 6 = 3: no measurement
S{5} = 0;   % k mod 6 = 4: no measurement
S{6} = 0;   % k mod 6 = 5: no measurement

% % Pattern 2: Sensor available every 3 steps (N=3 effective within N=6)
% S = cell(N, 1);
% S{1} = 1;   % k mod 6 = 0: measurement
% S{2} = 0;   % k mod 6 = 1: no measurement
% S{3} = 0;   % k mod 6 = 2: no measurement
% S{4} = 1;   % k mod 6 = 3: measurement
% S{5} = 0;   % k mod 6 = 4: no measurement
% S{6} = 0;   % k mod 6 = 5: no measurement

% % Pattern 3: Sensor available every 2 steps (N=2 effective within N=6)
% S = cell(N, 1);
% S{1} = 1;   % k mod 6 = 0: measurement
% S{2} = 0;   % k mod 6 = 1: no measurement
% S{3} = 1;   % k mod 6 = 2: measurement
% S{4} = 0;   % k mod 6 = 3: no measurement
% S{5} = 1;   % k mod 6 = 4: measurement
% S{6} = 0;   % k mod 6 = 5: no measurement

% % Pattern 4: Sensor available every step (standard Kalman filter, N=1 effective)
% S = cell(N, 1);
% for i = 1:N
%     S{i} = 1;
% end

fprintf('========================================\n');
fprintf('Simplified Multirate Kalman Filter\n');
fprintf('(LMI Design with Cyclic Reformulation)\n');
fprintf('========================================\n\n');

fprintf('System Parameters:\n');
fprintf('  State dimension: n = %d\n', n);
fprintf('  Output dimension: m = %d\n', m);
fprintf('  Period: N = %d\n', N);
fprintf('  System matrix A = %.3f (eigenvalue)\n', A);
fprintf('  Process noise Q = %.3f\n', Q);
fprintf('  Measurement noise R = %.3f\n\n', R);

% Display measurement pattern
fprintf('Measurement Pattern S_k:\n');
for i = 1:N
    fprintf('  S_%d = %d', i-1, S{i});
    if S{i} == 1
        fprintf(' (measurement available)\n');
    else
        fprintf(' (no measurement)\n');
    end
end
fprintf('\n');

%% 2. Cyclic Reformulation Construction
fprintf('=== Cyclic Reformulation ===\n');

% A_cyc: Nn × Nn cyclic matrix
% Structure: [0 0 ... 0 A; A 0 ... 0 0; 0 A ... 0 0; ...; 0 0 ... A 0]
A_cyc = zeros(N*n, N*n);
A_cyc(1:n, (N-1)*n+1:N*n) = A;  % First block row: (1,N) position
for i = 2:N
    A_cyc((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = A;  % Subdiagonal
end

% B_cyc: Nn × Np cyclic matrix (same structure as A_cyc)
B_cyc = zeros(N*n, N*p);
B_cyc(1:n, (N-1)*p+1:N*p) = B;
for i = 2:N
    B_cyc((i-1)*n+1:i*n, (i-2)*p+1:(i-1)*p) = B;
end

% C_cyc: Nm × Nn block-diagonal matrix
% C_cyc = diag(S_0*C, S_1*C, ..., S_{N-1}*C)
C_cyc = zeros(N*m, N*n);
for i = 1:N
    C_cyc((i-1)*m+1:i*m, (i-1)*n+1:i*n) = S{i}*C;
end

% Q_cyc: Nn × Nn (block diagonal with Q)
Q_cyc = kron(eye(N), Q);

% R_cyc: Nm × Nm (block diagonal with S_k*R*S_k')
% Note: When S_k = 0, the corresponding block is 0 (semidefinite!)
R_cyc = zeros(N*m, N*m);
for i = 1:N
    R_cyc((i-1)*m+1:i*m, (i-1)*m+1:i*m) = S{i} * R * S{i}';
end

fprintf('Cyclic System Dimensions:\n');
fprintf('  A_cyc: %d × %d\n', size(A_cyc));
fprintf('  B_cyc: %d × %d\n', size(B_cyc));
fprintf('  C_cyc: %d × %d\n', size(C_cyc));
fprintf('  Q_cyc: %d × %d\n', size(Q_cyc));
fprintf('  R_cyc: %d × %d\n\n', size(R_cyc));

% Display A_cyc matrix
fprintf('A_cyc matrix:\n');
disp(A_cyc);

% Display C_cyc matrix
fprintf('C_cyc matrix (diagonal shows S_k*C):\n');
disp(C_cyc);

% Display R_cyc diagonal
fprintf('R_cyc diagonal (shows S_k*R*S_k''):\n');
disp(diag(R_cyc)');

% Check R_cyc rank (key observation from paper)
rank_R = rank(R_cyc);
fprintf('R_cyc rank: %d / %d', rank_R, N*m);
if rank_R < N*m
    fprintf(' (SEMIDEFINITE - standard DARE cannot be used!)\n\n');
else
    fprintf(' (positive definite - standard DARE could be used)\n\n');
end

%% 3. Observability Check
O_cyc = obsv(A_cyc, C_cyc);
rank_O = rank(O_cyc);
fprintf('Observability Check:\n');
fprintf('  Observability matrix rank: %d / %d\n', rank_O, N*n);
if rank_O == N*n
    fprintf('  Result: Observable\n\n');
else
    fprintf('  Result: NOT Observable (filter may not work!)\n\n');
end

%% 4. LMI-based Filter Design (Dual LQR Formulation)
fprintf('=== LMI Optimization (Dual LQR) ===\n');

n_cyc = N*n;   % Cyclic state dimension
m_cyc = N*m;   % Cyclic output dimension

% Dual system matrices
Ad = A_cyc';   % Dual A
Bd = C_cyc';   % Dual B

% Q_cyc_sqrt: Cyclic structure (same pattern as A_cyc)
Q_sqrt = sqrt(Q);
Q_cyc_sqrt = zeros(n_cyc, n_cyc);
Q_cyc_sqrt(1:n, (N-1)*n+1:N*n) = Q_sqrt;
for i = 2:N
    Q_cyc_sqrt((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = Q_sqrt;
end

% R_cyc_sqrt: Block diagonal (with regularization for semidefinite case)
epsilon = 1e-8;
R_cyc_reg = R_cyc + epsilon*eye(m_cyc);
R_cyc_sqrt = zeros(m_cyc, m_cyc);
for i = 1:N
    R_block = S{i} * R * S{i}' + epsilon;
    R_cyc_sqrt((i-1)*m+1:i*m, (i-1)*m+1:i*m) = sqrt(R_block);
end

fprintf('Dual system:\n');
fprintf('  Ad = A_cyc'': %d × %d\n', size(Ad));
fprintf('  Bd = C_cyc'': %d × %d\n', size(Bd));
fprintf('  R_cyc regularization: epsilon = %.1e\n\n', epsilon);

%% LMI Setup
setlmis([]);

% Decision variables
X_var = lmivar(1, [n_cyc, 1]);      % X > 0 (Lyapunov matrix)
Y_var = lmivar(2, [m_cyc, n_cyc]);  % Y = -X*L (gain variable)
W_var = lmivar(1, [n_cyc, 1]);      % W >= X^{-1} (for trace minimization)

%% LMI #1: Stability and Performance (DARE-based LMI)
% [X,            (Ad*X+Bd*Y)',  X*Q_sqrt,    Y'*R_sqrt]
% [Ad*X+Bd*Y,    X,             0,           0        ]
% [Q_sqrt'*X,     0,             I,           0        ]
% [R_sqrt*Y,     0,             0,           I        ] >= 0

% Block (1,1): X
lmiterm([-1, 1, 1, X_var], 1, 1);

% Block (2,1): Ad*X + Bd*Y
lmiterm([-1, 2, 1, X_var], Ad, 1);
lmiterm([-1, 2, 1, Y_var], Bd, 1);

% Block (2,2): X
lmiterm([-1, 2, 2, X_var], 1, 1);

% Block (3,1): Q_sqrt*X
lmiterm([-1, 3, 1, X_var], Q_cyc_sqrt', 1);

% Block (3,3): I
lmiterm([-1, 3, 3, 0], eye(n_cyc));

% Block (4,1): R_sqrt*Y
lmiterm([-1, 4, 1, Y_var], R_cyc_sqrt', 1);

% Block (4,4): I
lmiterm([-1, 4, 4, 0], eye(m_cyc));

%% LMI #2: X > eps*I (numerical stability)
eps_val = 1e-6;
lmiterm([2, 1, 1, X_var], -1, 1);
lmiterm([2, 1, 1, 0], eps_val*eye(n_cyc));

%% LMI #3: [W, I; I, X] >= 0  =>  W >= X^{-1}
lmiterm([-3, 1, 1, W_var], 1, 1);
lmiterm([-3, 1, 2, 0], eye(n_cyc));
lmiterm([-3, 2, 2, X_var], 1, 1);

%% Solve LMI
lmisys = getlmis;

% Objective: minimize trace(W)
ndec = decnbr(lmisys);
c = zeros(ndec, 1);
for i = 1:ndec
    ei = zeros(ndec, 1);
    ei(i) = 1;
    W_i = dec2mat(lmisys, ei, W_var);
    c(i) = trace(W_i);
end

% Solve
options = [1e-5, 500, 1e9, 50, 1];
[copt, xopt] = mincx(lmisys, c, options);

if isempty(copt)
    error('LMI optimization failed!');
end

% Extract optimal matrices
X_opt = dec2mat(lmisys, xopt, X_var);
Y_opt = dec2mat(lmisys, xopt, Y_var);
W_opt = dec2mat(lmisys, xopt, W_var);

% Error covariance
P_ss = inv(X_opt);

% Kalman gain (from duality)
K_ss = -X_opt \ Y_opt;
K_ss = K_ss';  % Transpose for correct orientation

fprintf('LMI Results:\n');
fprintf('  Optimal trace(W) = %.6f (upper bound on trace(P))\n', copt);
fprintf('  Actual trace(P) = %.6f\n', trace(P_ss));
fprintf('  K_ss size: %d × %d\n\n', size(K_ss));

% Closed-loop stability check
A_cl = A_cyc - K_ss * C_cyc;
eig_cl = eig(A_cl);
max_eig = max(abs(eig_cl));
fprintf('Stability Analysis:\n');
fprintf('  Max |eigenvalue| of A_cyc - K_ss*C_cyc: %.6f\n', max_eig);
if max_eig < 1
    fprintf('  Result: STABLE\n\n');
else
    fprintf('  Result: UNSTABLE\n\n');
end

%% 5. Extract Periodic Kalman Gains
fprintf('=== Periodic Kalman Gains L_k ===\n');

L = cell(N, 1);
% Extract gains from cyclic structure
% L_0 is at block position (2,1) of K_ss
% L_k is at block position (k+2 mod N + 1, k+1) for k = 0,...,N-1
for k = 0:N-1
    row_block = mod(k+1, N) + 1;  % 1-indexed
    col_block = k + 1;
    L{k+1} = K_ss((row_block-1)*n+1:row_block*n, (col_block-1)*m+1:col_block*m);
end

fprintf('Periodic gains:\n');
for k = 0:N-1
    fprintf('  L_%d = %.6f', k, L{k+1});
    if S{k+1} == 1
        fprintf('  (measurement available)\n');
    else
        fprintf('  (no measurement, gain has no effect)\n');
    end
end
fprintf('\n');

%% 6. Simulation
fprintf('=== Simulation ===\n');

T = 100;   % Simulation length
rng(42);   % For reproducibility

% True state
x_true = zeros(1, T);
x_true(1) = 5;  % Initial state

% Observations
y_obs = zeros(1, T);

% Control input
u = 0.5*sin(0.1*(0:T-1));

% Generate true trajectory
for k = 1:T-1
    w = sqrt(Q) * randn;
    x_true(k+1) = A * x_true(k) + B * u(k) + w;
end

% Generate observations
for k = 1:T
    v = sqrt(R) * randn;
    y_obs(k) = C * x_true(k) + v;
end

% Kalman filter estimation
x_hat = zeros(1, T);
x_hat(1) = x_true(1);  % Perfect initial condition

for k = 1:T-1
    % Prediction
    x_pred = A * x_hat(k) + B * u(k);
    y_pred = C * x_pred;
    
    % Update with periodic gain
    idx = mod(k-1, N) + 1;  % k=1 corresponds to S{1}
    S_k = S{idx};
    
    if S_k == 1
        % Measurement available
        innovation = y_obs(k+1) - y_pred;
        x_hat(k+1) = x_pred + L{idx} * innovation;
    else
        % No measurement
        x_hat(k+1) = x_pred;
    end
end

% Performance metrics
error = x_true - x_hat;
rmse = sqrt(mean(error.^2));
max_error = max(abs(error));

fprintf('Performance:\n');
fprintf('  RMSE: %.4f\n', rmse);
fprintf('  Max error: %.4f\n\n', max_error);

%% 7. Plots
figure('Position', [100 100 1400 800]);

% State estimation
subplot(2,3,1);
plot(0:T-1, x_true, 'k-', 'LineWidth', 2); hold on;
plot(0:T-1, x_hat, 'b--', 'LineWidth', 1.5);
% Mark observation times
obs_times = find(cellfun(@(s) s==1, S(mod(0:T-1, N)+1)));
plot(obs_times-1, y_obs(obs_times), 'ro', 'MarkerSize', 6);
xlabel('Time step k'); ylabel('State x');
title('State Estimation');
legend({'True', 'Estimated', 'Observations'}, 'Location', 'best');
grid on;

% Estimation error
subplot(2,3,2);
plot(0:T-1, error, 'LineWidth', 1.5);
yline(0, 'k--');
xlabel('Time step k'); ylabel('Error');
title(sprintf('Estimation Error (RMSE = %.4f)', rmse));
grid on;

% Measurement pattern
subplot(2,3,3);
pattern = zeros(1, T);
for k = 1:T
    pattern(k) = S{mod(k-1, N)+1};
end
stem(0:T-1, pattern, 'filled', 'MarkerSize', 3);
xlabel('Time step k'); ylabel('S_k');
title('Measurement Availability Pattern');
ylim([-0.1, 1.1]);
grid on;

% Closed-loop eigenvalues
subplot(2,3,4);
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1); hold on;
plot(real(eig_cl), imag(eig_cl), 'bo', 'MarkerSize', 10, 'LineWidth', 2);
axis equal; grid on;
xlabel('Real'); ylabel('Imaginary');
title(sprintf('Closed-loop Eigenvalues (max|λ| = %.4f)', max_eig));
legend({'Unit circle', 'Eigenvalues'}, 'Location', 'best');

% Periodic gains
subplot(2,3,5);
L_vals = cellfun(@(x) x(1), L);
bar(0:N-1, L_vals);
xlabel('k mod N'); ylabel('L_k');
title('Periodic Kalman Gains');
grid on;

% K_ss matrix visualization
subplot(2,3,6);
imagesc(K_ss);
colorbar;
xlabel('Column (output block)'); ylabel('Row (state block)');
title('K_{ss} Matrix Structure');
axis equal tight;

sgtitle(sprintf('Multirate Kalman Filter: n=%d, m=%d, N=%d', n, m, N));

%% 8. Compare with Standard Kalman Filter (if all measurements available)
fprintf('=== Comparison with Standard Kalman Filter ===\n');

% Standard DARE solution (assuming measurement at every step)
P_std = dare(A', C', Q, R);
K_std = A * P_std * C' / (C * P_std * C' + R);

fprintf('Standard Kalman filter (all S_k = 1):\n');
fprintf('  Steady-state gain: K = %.6f\n', K_std);
fprintf('  Steady-state covariance: P = %.6f\n\n', P_std);

% Simulation with standard KF
x_hat_std = zeros(1, T);
x_hat_std(1) = x_true(1);
for k = 1:T-1
    x_pred = A * x_hat_std(k) + B * u(k);
    y_pred = C * x_pred;
    innovation = y_obs(k+1) - y_pred;
    x_hat_std(k+1) = x_pred + K_std * innovation;
end

error_std = x_true - x_hat_std;
rmse_std = sqrt(mean(error_std.^2));

fprintf('Performance comparison:\n');
fprintf('  Multirate KF RMSE: %.4f\n', rmse);
fprintf('  Standard KF RMSE (if all obs available): %.4f\n', rmse_std);

% Count actual observations in multirate case
n_obs = sum(cellfun(@(s) s==1, S(mod(0:T-1, N)+1)));
fprintf('  Observations used: %d / %d (%.1f%%)\n', n_obs, T, 100*n_obs/T);

fprintf('\n========================================\n');
fprintf('Program completed successfully\n');
fprintf('========================================\n');

%% Helper function
function result = iif(cond, true_val, false_val)
    if cond
        result = true_val;
    else
        result = false_val;
    end
end
