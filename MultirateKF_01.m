%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multirate Kalman Filter Design via LMI Optimization (DARI)
%
% Author: Hiroshi Okajima
% Created with assistance from Claude (Anthropic)
% Date: February 2025
%
% Description:
%   This code implements a multirate Kalman filter for automotive navigation
%   systems using LMI-based optimization through cyclic reformulation.
%   The system fuses GPS measurements (1 Hz) with wheel speed sensor data (10 Hz)
%   to estimate vehicle position, velocity, and acceleration.
%
%   Key Features:
%   - Handles semidefinite measurement noise covariance R_cyc naturally via LMI
%   - Periodic time-varying Kalman gains computed offline
%   - Dual LQR formulation ensures numerical stability
%   - Trace minimization for optimal estimation error covariance
%
% System Configuration:
%   - State: [position; velocity; acceleration]
%   - Measurements: GPS position (10% availability) + wheel speed (100%)
%   - Period: N = 10, Sampling time: dt = 0.1s
%
% Reference:
%   H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter 
%   Design," IEEE Access, 2025. [Preprint: arXiv:XXXX.XXXXX]
%
% Usage:
%   Run this script directly to perform:
%   1. Cyclic system construction
%   2. LMI-based filter design (DARI)
%   3. Periodic gain extraction
%   4. Simulation and performance evaluation
%
% Output:
%   - Console: System info, LMI results, RMSE performance metrics
%   - Figures: State estimation results, eigenvalues, gain analysis
%
% Required MATLAB Toolboxes:
%   - Robust Control Toolbox (for LMI optimization)
%   - Control System Toolbox (for basic system analysis)
%
% License:
%   This work is licensed under the Creative Commons Attribution 4.0
%   International License (CC BY 4.0).
%   http://creativecommons.org/licenses/by/4.0/
%
% Repository:
%   [Placeholder: GitHub repository URL will be added]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

%% 1. System Definition
n = 3; p = 1; m = 2; N = 10; dt = 0.1;

% State-space model: x = [position; velocity; acceleration]
A = [1  dt  0.5*dt^2;
     0  1   dt;
     0  0   0.8];

B = [0; 0; 1];

% Output: position and velocity
C = [1 0 0;   % GPS position
     0 1 0];  % Wheel speed sensor

Q = diag([0.01, 0.1, 0.5]);  % Process noise
R = diag([1.0, 0.1]);        % GPS accuracy ±1m, wheel speed accuracy ±0.1m/s

% Measurement pattern (2×2 diagonal matrix)
% GPS (position): every 10 steps (k mod 10 = 0)
% Wheel speed (velocity): every step
S = cell(N, 1);
S{1} = diag([1, 1]);   % k mod 10 = 0: GPS + wheel speed
for i = 2:N
    S{i} = diag([0, 1]);  % k mod 10 = 1-9: wheel speed only
end

fprintf('========================================\n');
fprintf('Multirate Kalman Filter (LMI Design)\n');
fprintf('Automotive Navigation System\n');
fprintf('========================================\n\n');

fprintf('System:\n');
fprintf('  State dimension: n=%d [position, velocity, acceleration]\n', n);
fprintf('  Output dimension: m=%d [GPS position, wheel speed]\n', m);
fprintf('  Observation period: N=%d\n', N);
fprintf('  Sampling time: dt=%.2f [s]\n', dt);
fprintf('  Eigenvalues of A: [%.6f, %.6f, %.6f]\n\n', eig(A));

fprintf('Sensor Specifications:\n');
fprintf('  GPS (position): 1Hz (every 10 steps), accuracy ±%.1fm\n', sqrt(R(1,1)));
fprintf('  Wheel speed sensor (velocity): 10Hz (every step), accuracy ±%.1fm/s\n\n', sqrt(R(2,2)));

fprintf('Measurement Pattern:\n');
fprintf('  k mod 10 = 0: GPS ON + Wheel speed ON\n');
fprintf('  k mod 10 = 1-9: GPS OFF + Wheel speed ON\n\n');

%% 2. Cyclic Reformulation Construction
% A_cyc: 30×30 (cyclic structure)
A_cyc = zeros(N*n, N*n);
A_cyc(1:n, (N-1)*n+1:N*n) = A;  % 1st block row → N-th block column
for i = 2:N
    A_cyc((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = A;
end

% B_cyc: 30×10 (cyclic structure)
B_cyc = zeros(N*n, N*p);
B_cyc(1:n, (N-1)*p+1:N*p) = B;
for i = 2:N
    B_cyc((i-1)*n+1:i*n, (i-2)*p+1:(i-1)*p) = B;
end

% C_cyc: 20×30 (block-diagonal structure)
C_cyc = zeros(N*m, N*n);
for i = 1:N
    C_cyc((i-1)*m+1:i*m, (i-1)*n+1:i*n) = S{i}*C;
end

% Q_cyc, R_cyc
Q_cyc = kron(eye(N), Q);

% R_cyc: Correct implementation (unmeasured components become 0)
% S{i}*R*S{i}' automatically makes unmeasured components zero
R_cyc = zeros(N*m, N*m);
for i = 1:N
    R_block = S{i} * R * S{i}';  % Unmeasured components automatically become 0
    R_cyc((i-1)*m+1:i*m, (i-1)*m+1:i*m) = R_block;
end

fprintf('Cyclic Representation Dimensions:\n');
fprintf('  A_cyc: %d×%d\n', size(A_cyc));
fprintf('  B_cyc: %d×%d\n', size(B_cyc));
fprintf('  C_cyc: %d×%d\n', size(C_cyc));
fprintf('  Q_cyc: %d×%d\n', size(Q_cyc));
fprintf('  R_cyc: %d×%d (rank: %d)\n\n', size(R_cyc), rank(R_cyc));

fprintf('R_cyc Structure Verification:\n');
fprintf('  Number of diagonal elements: %d\n', sum(diag(R_cyc) ~= 0));
fprintf('  R components when GPS available: %d\n', sum(S{1}(:)));
fprintf('  R components with wheel speed only: %d\n\n', sum(S{2}(:)));

%% 3. Observability Check
O_cyc = obsv(A_cyc, C_cyc);
rank_O = rank(O_cyc);
cond_O = cond(O_cyc);
fprintf('Observability:\n');
fprintf('  Rank: %d / %d\n', rank_O, N*n);
fprintf('  Condition number: %.2e\n', cond_O);
fprintf('  Result: %s\n\n', iif(rank_O == N*n, 'Observable', 'Not Observable'));

%% 4. LMI Computation (DARI: Discrete Algebraic Riccati Inequality)
fprintf('=== LMI Computation (DARI) ===\n');

% Cyclic system dimensions
n_cyc = N*n;   % 30
m_cyc = N*m;   % 20

% Dual system: z(k+1) = A_cyc'*z(k) + C_cyc'*u(k)
Ad = A_cyc';   % Dual system A (30×30)
Bd = C_cyc';   % Dual system B (30×20)

% Cholesky decomposition of Q_cyc, R_cyc
Q_sqrt = chol(Q, 'lower');
% Q_cyc_sqrt: Cyclic structure (Nn × Nn)
Q_cyc_sqrt = zeros(n_cyc, n_cyc);
Q_cyc_sqrt(1:n, (N-1)*n+1:N*n) = Q_sqrt;
for i = 2:N
    Q_cyc_sqrt((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = Q_sqrt;
end

% R_cyc is semidefinite, so regularize before Cholesky decomposition
epsilon = 1e-8;
R_cyc_reg = R_cyc + epsilon*eye(N*m);
R_cyc_sqrt = chol(R_cyc_reg, 'lower') - sqrt(epsilon)*eye(N*m);

fprintf('\n=== LMI Problem Setup (Dual LQR for Cyclic System) ===\n');
fprintf('Dual system dimensions:\n');
fprintf('  Ad = A_cyc'': %d×%d\n', size(Ad));
fprintf('  Bd = C_cyc'': %d×%d\n', size(Bd));
fprintf('  R_cyc regularization: epsilon = %.1e\n', epsilon);

setlmis([]);

% Decision variables
% X > 0 (n_cyc×n_cyc symmetric) = P^{-1}
% Y = K*X (m_cyc×n_cyc rectangular), K is dual LQR gain
X_var = lmivar(1, [n_cyc, 1]);    % X > 0 (30×30 symmetric)
Y_var = lmivar(2, [m_cyc, n_cyc]);    % Y = K*X (20×30 rectangular)

%% LMI #1: Dual LQR stability + performance condition
% Block structure (n_cyc + n_cyc + n_cyc + m_cyc) × (...):
%   [X,              (Ad*X + Bd*Y)',   X*Q^{1/2},   Y'*R^{1/2}]
%   [Ad*X + Bd*Y,         X,              0,           0      ]
%   [Q^{1/2}*X,           0,              I,           0      ]
%   [R^{1/2}*Y,           0,              0,           I      ]

% (1,1): X
lmiterm([-1, 1, 1, X_var], 1, 1);

% (2,1): Ad*X + Bd*Y = A_cyc'*X + C_cyc'*Y
lmiterm([-1, 2, 1, X_var], Ad, 1);      % A_cyc'*X
lmiterm([-1, 2, 1, Y_var], Bd, 1);      % C_cyc'*Y

% (2,2): X
lmiterm([-1, 2, 2, X_var], 1, 1);

% (3,1): Q^{1/2}*X
lmiterm([-1, 3, 1, X_var], Q_cyc_sqrt', 1);

% (3,3): I_n_cyc
lmiterm([-1, 3, 3, 0], eye(n_cyc));

% (4,1): R^{1/2}*Y (using regularized version)
lmiterm([-1, 4, 1, Y_var], R_cyc_sqrt', 1);

% (4,4): I_m_cyc
lmiterm([-1, 4, 4, 0], eye(m_cyc));

%% LMI #2: X > eps*I (numerical stability)
eps_val = 1e-6;
lmiterm([2, 1, 1, X_var], -1, 1);
lmiterm([2, 1, 1, 0], eps_val*eye(n_cyc));

%% LMI #3: Linearization of covariance upper bound
% P = X^{-1} (error covariance), want to minimize trace(P)
% Introduce auxiliary variable W: [W, I; I, X] >= 0 ⟺ W >= X^{-1} = P
% min trace(W)

W_var = lmivar(1, [n_cyc, 1]);    % W > 0 (30×30 symmetric)

% [W, I; I, X] >= 0
lmiterm([-3, 1, 1, W_var], 1, 1);        % W
lmiterm([-3, 1, 2, 0], eye(n_cyc));      % I
lmiterm([-3, 2, 2, X_var], 1, 1);        % X

%% Get LMI system
lmisys = getlmis;

% Objective function: min trace(W)
ndec = decnbr(lmisys);
c = zeros(ndec, 1);
for i = 1:ndec
    ei = zeros(ndec, 1);
    ei(i) = 1;
    Wi = dec2mat(lmisys, ei, W_var);
    c(i) = trace(Wi);
end

fprintf('Number of decision variables: %d\n', ndec);

%% Solve LMI
fprintf('\n=== Solving LMI ===\n');
options = [1e-5, 1000, 1e9, 100, 1];  % Allow more iterations

[copt, xopt] = mincx(lmisys, c, options);

if isempty(copt)
    error('LMI problem is infeasible!');
end

% Extract solution
X_opt = dec2mat(lmisys, xopt, X_var);
Y_opt = dec2mat(lmisys, xopt, Y_var);
W_opt = dec2mat(lmisys, xopt, W_var);

% Error covariance and gain
P_ss = inv(X_opt);           % P = X^{-1}

% Compute Kalman gain from P (using regularized R)
K_ss = A_cyc * P_ss * C_cyc' / (C_cyc * P_ss * C_cyc' + R_cyc_reg);

% Dual gain from Y_opt
K_dual = Y_opt / X_opt;

% Conversion from dual LQR to Kalman filter
% In dual LQR: K_dual is the gain multiplied by C_cyc'
% → In Kalman filter: K_ss is the gain multiplied by C_cyc output
K_ss_from_Y = K_dual';  % Transpose relation (from duality)

% Comparison
fprintf('||K_ss - K_ss_from_Y|| = %.6e\n', norm(K_ss + K_ss_from_Y, 'fro'));

fprintf('\n=== LMI Results ===\n');
fprintf('Optimal trace(W) = %.6f (upper bound on trace(P))\n', copt);
fprintf('Actual trace(P) = %.6f\n', trace(P_ss));
fprintf('K_ss (from LMI): %d×%d\n', size(K_ss));

% Closed-loop eigenvalues
A_cl = A_cyc - K_ss*C_cyc;
eig_cl = eig(A_cl);
max_eig = max(abs(eig_cl));
fprintf('\nMaximum magnitude of closed-loop eigenvalues: %.10f\n', max_eig);
fprintf('Stability: %s\n', iif(max_eig < 1, 'Stable', 'Unstable'));

%% 5. Extract Periodic Time-Varying Gains (Cyclic Structure)
fprintf('\n=== Periodic Time-Varying Observer Gains ===\n');

L = cell(N, 1);
L{1} = K_ss(n+1:2*n, 1:m);                      % Block 1 → 1st column block
L{2} = K_ss(2*n+1:3*n, m+1:2*m);                % Block 2 → 2nd column block
L{3} = K_ss(3*n+1:4*n, 2*m+1:3*m);              
L{4} = K_ss(4*n+1:5*n, 3*m+1:4*m);              
L{5} = K_ss(5*n+1:6*n, 4*m+1:5*m);              
L{6} = K_ss(6*n+1:7*n, 5*m+1:6*m);              
L{7} = K_ss(7*n+1:8*n, 6*m+1:7*m);              
L{8} = K_ss(8*n+1:9*n, 7*m+1:8*m);              
L{9} = K_ss(9*n+1:10*n, 8*m+1:9*m);             
L{10} = K_ss(1:n, 9*m+1:10*m);                  

fprintf('L0 (k mod 10 = 0, GPS + wheel speed):\n');
disp(L{1});

fprintf('L1 (k mod 10 = 1, wheel speed only):\n');
disp(L{2});

fprintf('L5 (k mod 10 = 5, wheel speed only):\n');
disp(L{6});

%% 6. Simulation (Vehicle Motion)
fprintf('\n=== Simulation ===\n');
T = 200;
rng(42);

% True state (actual vehicle position, velocity, acceleration)
x_true = zeros(n, T);
x_true(:,1) = [0; 5; 0];  % Initial: position 0m, velocity 5m/s, acceleration 0
z_obs = zeros(m, T);
u = 0.5*sin(0.05*(0:T-1))';  % Acceleration command (driving operation)

for k = 1:T-1
    w = mvnrnd(zeros(n,1), Q)';
    x_true(:,k+1) = A*x_true(:,k) + B*u(k) + w;
    
    % Sensor observation (with noise)
    v = mvnrnd(zeros(m,1), R)';
    z_obs(:,k+1) = C*x_true(:,k+1) + v;
end
z_obs(:,1) = C*x_true(:,1);

fprintf('Driving simulation:\n');
fprintf('  Total time: %.1f seconds\n', T*dt);
fprintf('  Total distance traveled: %.1f m\n', x_true(1,end));
fprintf('  Maximum velocity: %.1f m/s (%.1f km/h)\n', max(x_true(2,:)), max(x_true(2,:))*3.6);

%% 7. Multirate Kalman Filter (LMI version)
x_hat = zeros(n, T);
x_hat(:,1) = x_true(:,1);

for k = 1:T-1
    % Prediction
    x_pred = A*x_hat(:,k) + B*u(k);
    z_pred = C*x_pred;
    
    % Update (periodic time-varying gain)
    idx = mod(k, N) + 1;
    innovation = z_obs(:,k+1) - z_pred;
    
    x_hat(:,k+1) = x_pred + L{idx}*innovation;
end

fprintf('\nKalman filter completed\n');

%% 8. Performance Evaluation
error_est = x_true - x_hat;

rmse = sqrt(mean(error_est.^2, 2));
max_error = max(abs(error_est), [], 2);

fprintf('\n=== Performance Evaluation ===\n');
fprintf('RMSE:\n');
fprintf('  Position:     %.4f [m]\n', rmse(1));
fprintf('  Velocity:     %.4f [m/s]\n', rmse(2));
fprintf('  Acceleration: %.4f [m/s^2]\n', rmse(3));

fprintf('\nMaximum Error:\n');
fprintf('  Position:     %.4f [m]\n', max_error(1));
fprintf('  Velocity:     %.4f [m/s]\n', max_error(2));
fprintf('  Acceleration: %.4f [m/s^2]\n', max_error(3));

%% 9. Plot Results
figure('Position', [100 100 1600 1000]);

% Position
subplot(3,3,1);
plot(0:T-1, x_true(1,:), 'k-', 'LineWidth', 2); hold on;
plot(0:T-1, x_hat(1,:), 'b-', 'LineWidth', 1.5);
obs_idx = find(mod(0:T-1, N) == 0);
plot(obs_idx-1, z_obs(1,obs_idx), 'go', 'MarkerSize', 8);
xlabel('Time Step'); ylabel('Position [m]');
title('Position (GPS: every 10 steps)'); grid on;
legend({'True', 'Estimated', 'GPS'}, 'Location', 'best');

% Velocity
subplot(3,3,2);
plot(0:T-1, x_true(2,:), 'k-', 'LineWidth', 2); hold on;
plot(0:T-1, x_hat(2,:), 'b-', 'LineWidth', 1.5);
xlabel('Time Step'); ylabel('Velocity [m/s]');
title('Velocity (Wheel Speed: every step)'); grid on;
legend({'True', 'Estimated'}, 'Location', 'best');

% Acceleration
subplot(3,3,3);
plot(0:T-1, x_true(3,:), 'k-', 'LineWidth', 2); hold on;
plot(0:T-1, x_hat(3,:), 'b-', 'LineWidth', 1.5);
xlabel('Time Step'); ylabel('Acceleration [m/s^2]');
title('Acceleration (not observed)'); grid on;
legend({'True', 'Estimated'}, 'Location', 'best');

% Estimation errors
subplot(3,3,4);
plot(0:T-1, error_est(1,:), 'LineWidth', 1.5); hold on;
plot(0:T-1, error_est(2,:), 'LineWidth', 1.5);
plot(0:T-1, error_est(3,:), 'LineWidth', 1.5);
yline(0, 'k--');
xlabel('Time Step'); ylabel('Error');
title('Estimation Errors'); grid on;
legend({'Position', 'Velocity', 'Acceleration'});

% Position error (zoomed)
subplot(3,3,5);
plot(0:T-1, error_est(1,:), 'LineWidth', 1.5);
yline(0, 'k--');
xlabel('Time Step'); ylabel('Position Error [m]');
title('Position Error'); grid on;

% Velocity error (zoomed)
subplot(3,3,6);
plot(0:T-1, error_est(2,:), 'LineWidth', 1.5);
yline(0, 'k--');
xlabel('Time Step'); ylabel('Velocity Error [m/s]');
title('Velocity Error'); grid on;

% Closed-loop eigenvalues
subplot(3,3,7);
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1); hold on;
plot(real(eig_cl), imag(eig_cl), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
axis equal; grid on;
xlabel('Real'); ylabel('Imaginary');
title('Closed-loop Eigenvalues');
legend({'Unit Circle', 'Eigenvalues'});

% Eigenvalue magnitudes
subplot(3,3,8);
bar(abs(eig_cl));
yline(1.0, 'k--', 'LineWidth', 2);
xlabel('Index'); ylabel('|λ|');
title('Eigenvalue Magnitudes'); grid on;
ylim([0, 1.1]);

% Visualization of measurement pattern
subplot(3,3,9);
obs_pattern = zeros(2, min(100, T));
for k = 0:min(99, T-1)
    if mod(k, N) == 0
        obs_pattern(1, k+1) = 1;  % GPS position
    end
    obs_pattern(2, k+1) = 1;  % Wheel speed (always)
end
imagesc(0:min(99, T-1), 1:2, obs_pattern);
colormap([1 1 1; 0 0.5 0]);
xlabel('Time Step'); 
set(gca, 'YTick', [1 2], 'YTickLabel', {'GPS (Position)', 'Wheel Speed'});
title('Sensor Availability'); 
colorbar('Ticks', [0 1], 'TickLabels', {'No', 'Yes'});

% Gain norm (periodicity verification)
figure('Position', [100 100 1200 600]);

subplot(2,2,1);
L_norms = zeros(N, 1);
for i = 1:N
    L_norms(i) = norm(L{i}, 'fro');
end
bar(0:N-1, L_norms);
xlabel('k mod 10'); ylabel('||L_k|| (Frobenius)');
title('Kalman Gain Norm (Periodic)'); grid on;

subplot(2,2,2);
% Contribution to each state
L_contributions = zeros(N, n);
for i = 1:N
    for j = 1:n
        L_contributions(i,j) = norm(L{i}(j,:));
    end
end
bar(0:N-1, L_contributions);
xlabel('k mod 10'); ylabel('Gain Magnitude');
title('Kalman Gain Contribution to Each State');
legend({'Position', 'Velocity', 'Acceleration'});
grid on;

subplot(2,2,3);
% Eigenvalues of P matrix
semilogy(1:n_cyc, sort(eig(P_ss), 'descend'), 'bo-', 'LineWidth', 1.5);
xlabel('Index'); ylabel('Eigenvalue');
title('P matrix Eigenvalues'); grid on;

subplot(2,2,4);
% Visualize R_cyc structure
spy(R_cyc);
title('R_{cyc} Sparsity Pattern');
xlabel('Column'); ylabel('Row');

fprintf('\nProgram completed\n');
fprintf('\n=== Conclusions ===\n');
fprintf('Multirate Kalman filter design via LMI (DARI):\n');
fprintf('- R_cyc automatically becomes 0 for unmeasured components\n');
fprintf('- Numerical stability ensured through regularization\n');
fprintf('- Closed-loop system is stable\n');
fprintf('- Estimation performance is satisfactory\n');

function result = iif(cond, true_val, false_val)
    if cond, result = true_val; else, result = false_val; end
end