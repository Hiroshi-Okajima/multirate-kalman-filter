%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multirate Kalman Filter - Kalman Filter/Eigenvalue Placement Multi-Objective Design
%
% Author: Hiroshi Okajima
% Created with assistance from Claude (Anthropic)
% Date: February 2025
%
% Description:
%   This code implements multi-objective design combining optimal Kalman filtering
%   and eigenvalue placement constraints for multirate systems.
%   The unified LMI framework handles both objectives simultaneously:
%   - Optimal Kalman filter: Minimizes estimation error covariance
%   - Eigenvalue placement: Guarantees convergence rate |λ| < r_bar
%
% Unified LMI Structure:
%   [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
%   [*,    (2,2),   0,         0       ]
%   [*,    0,       (3,3),     0       ]
%   [*,    0,       0,         (4,4)   ] > 0
%
%   DARI (Optimal Kalman): (2,2)=X,    (3,3)=I,    (4,4)=I
%   Eigenvalue constraint:  (2,2)=r²X  (all eigenvalues within radius r)
%
% Common variables: X, Y, W
% Y = -XL formulation (X is Lyapunov matrix, X = P^{-1})
% Objective: min trace(W) where W >= X^{-1} = P
%
% Reference:
%   H. Okajima, "LMI Optimization Based Multirate Steady-State Kalman Filter 
%   Design," IEEE Access, 2025. [Preprint: arXiv:XXXX.XXXXX]
%
% Usage:
%   Run this script directly to:
%   1. Compute optimal Kalman filter design (baseline)
%   2. Sweep eigenvalue radius constraints (r_bar = 0.975 to 0.75)
%   3. Analyze trade-off between performance and convergence rate
%   4. Compare gains for different designs
%
% Output:
%   - Console: Trade-off results, gain comparisons
%   - Figures: trace(W) vs r_bar, max|eig| vs r_bar
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

A = [1  dt  0.5*dt^2;
     0  1   dt;
     0  0   0.8];

B = [0; 0; 1];

C = [1 0 0;
     0 1 0];

Q = diag([0.01, 0.1, 0.5]);
R = diag([1.0, 0.1]);

% Measurement pattern
S_mat = cell(N, 1);
S_mat{1} = diag([1, 1]);
for i = 2:N
    S_mat{i} = diag([0, 1]);
end

fprintf('========================================\n');
fprintf('Kalman Filter/Eigenvalue Placement Multi-Objective Design\n');
fprintf('========================================\n\n');

%% 2. Cyclic Reformulation Construction
n_cyc = N*n;
m_cyc = N*m;

% A_cyc: Cyclic structure
A_cyc = zeros(n_cyc, n_cyc);
A_cyc(1:n, (N-1)*n+1:N*n) = A;
for i = 2:N
    A_cyc((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = A;
end

% C_cyc: Block-diagonal structure (S_k * C)
C_cyc = zeros(m_cyc, n_cyc);
for i = 1:N
    C_cyc((i-1)*m+1:i*m, (i-1)*n+1:i*n) = S_mat{i}*C;
end

% Q^{1/2}: Cyclic structure
Q_sqrt = sqrtm(Q);
Q_cyc_sqrt = zeros(n_cyc, n_cyc);
Q_cyc_sqrt(1:n, (N-1)*n+1:N*n) = Q_sqrt;
for i = 2:N
    Q_cyc_sqrt((i-1)*n+1:i*n, (i-2)*n+1:(i-1)*n) = Q_sqrt;
end

% R^{1/2}: Block-diagonal (reflecting measurement pattern)
R_sqrt = sqrtm(R);
R_cyc_sqrt = zeros(m_cyc, m_cyc);
for i = 1:N
    R_cyc_sqrt((i-1)*m+1:i*m, (i-1)*m+1:i*m) = S_mat{i}*R_sqrt;
end

fprintf('System:\n');
fprintf('  State dimension: n=%d, Output dimension: m=%d, Period: N=%d\n', n, m, N);
fprintf('  n_cyc=%d, m_cyc=%d\n\n', n_cyc, m_cyc);

options = [1e-5, 1000, 1e9, 100, 1];
eps_val = 1e-6;

%% 3. Optimal Kalman Filter (DARI only) - Baseline
fprintf('=== Optimal Kalman Filter (DARI only) ===\n');

setlmis([]);

[X_var_kf, ~, ~] = lmivar(1, [n_cyc 1]);
[Y_var_kf, ~, ~] = lmivar(2, [n_cyc m_cyc]);
[W_var_kf, ~, ~] = lmivar(1, [n_cyc 1]);

% DARI LMI
lmi_dari_kf = newlmi;
lmiterm([-lmi_dari_kf, 1, 1, X_var_kf], 1, 1);
lmiterm([-lmi_dari_kf, 1, 2, X_var_kf], 1, A_cyc);
lmiterm([-lmi_dari_kf, 1, 2, Y_var_kf], 1, C_cyc);
lmiterm([-lmi_dari_kf, 2, 2, X_var_kf], 1, 1);
lmiterm([-lmi_dari_kf, 1, 3, X_var_kf], 1, Q_cyc_sqrt);
lmiterm([-lmi_dari_kf, 3, 3, 0], eye(n_cyc));
lmiterm([-lmi_dari_kf, 1, 4, Y_var_kf], 1, R_cyc_sqrt);
lmiterm([-lmi_dari_kf, 4, 4, 0], eye(m_cyc));

% X > eps*I
lmi_pos_kf = newlmi;
lmiterm([lmi_pos_kf, 1, 1, X_var_kf], -1, 1);
lmiterm([lmi_pos_kf, 1, 1, 0], eps_val*eye(n_cyc));

% [W, I; I, X] >= 0
lmi_cov_kf = newlmi;
lmiterm([-lmi_cov_kf, 1, 1, W_var_kf], 1, 1);
lmiterm([-lmi_cov_kf, 1, 2, 0], eye(n_cyc));
lmiterm([-lmi_cov_kf, 2, 2, X_var_kf], 1, 1);

lmisys_kf = getlmis;

ndec_kf = decnbr(lmisys_kf);
c_kf = zeros(ndec_kf, 1);
for i = 1:ndec_kf
    ei = zeros(ndec_kf, 1);
    ei(i) = 1;
    Wi = dec2mat(lmisys_kf, ei, W_var_kf);
    c_kf(i) = trace(Wi);
end

[copt_kf, xopt_kf] = mincx(lmisys_kf, c_kf, options);

if isempty(copt_kf)
    error('Optimal Kalman filter design is infeasible');
end

X_kf = dec2mat(lmisys_kf, xopt_kf, X_var_kf);
Y_kf = dec2mat(lmisys_kf, xopt_kf, Y_var_kf);
L_kf = -X_kf \ Y_kf;

A_cl_kf = A_cyc - L_kf * C_cyc;
eig_kf = eig(A_cl_kf);

fprintf('Optimal Kalman filter solution:\n');
fprintf('  trace(W) = %.4f\n', copt_kf);
fprintf('  trace(X^{-1}) = %.4f\n', trace(inv(X_kf)));
fprintf('  max|eig| = %.6f\n\n', max(abs(eig_kf)));

%% 4. Kalman Filter/Eigenvalue Placement Multi-Objective Design
fprintf('=== Kalman Filter/Eigenvalue Placement Multi-Objective Design ===\n');
fprintf('LMI #1 (DARI):      (2,2)=X\n');
fprintf('LMI #2 (Eigenvalue): (1,1)=r²X, (2,2)=X  →  |λ| < r\n');
fprintf('Common variables: X, Y\n');
fprintf('Objective: min trace(W) where W >= X^{-1}\n\n');

% Vary radius from 0.975 to 0.75 in steps of 0.025
r_bar_list = 0.975:-0.025:0.75;

results = struct();
results.r_bar = [];
results.trace_W = [];
results.trace_Xinv = [];
results.max_eig = [];
results.feasible = [];

for idx = 1:length(r_bar_list)
    r_bar = r_bar_list(idx);
    
    fprintf('r_bar = %.3f: ', r_bar);
    
    setlmis([]);
    
    % Common variables
    [X_var_mix, ~, ~] = lmivar(1, [n_cyc 1]);
    [Y_var_mix, ~, ~] = lmivar(2, [n_cyc m_cyc]);
    [W_var_mix, ~, ~] = lmivar(1, [n_cyc 1]);
    
    % === LMI #1: DARI (Kalman filter performance) ===
    % [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
    % [*,    X,       0,         0       ]
    % [*,    0,       I,         0       ]
    % [*,    0,       0,         I       ] > 0
    lmi_dari_mix = newlmi;
    lmiterm([-lmi_dari_mix, 1, 1, X_var_mix], 1, 1);
    lmiterm([-lmi_dari_mix, 1, 2, X_var_mix], 1, A_cyc);
    lmiterm([-lmi_dari_mix, 1, 2, Y_var_mix], 1, C_cyc);
    lmiterm([-lmi_dari_mix, 2, 2, X_var_mix], 1, 1);
    lmiterm([-lmi_dari_mix, 1, 3, X_var_mix], 1, Q_cyc_sqrt);
    lmiterm([-lmi_dari_mix, 3, 3, 0], eye(n_cyc));
    lmiterm([-lmi_dari_mix, 1, 4, Y_var_mix], 1, R_cyc_sqrt);
    lmiterm([-lmi_dari_mix, 4, 4, 0], eye(m_cyc));
    
    % === LMI #2: Eigenvalue constraint (|λ| < r_bar) ===
    % [r²X, XA+YC; *, X] > 0
    lmi_eig_mix = newlmi;
    lmiterm([-lmi_eig_mix, 1, 1, X_var_mix], r_bar^2, 1);
    lmiterm([-lmi_eig_mix, 1, 2, X_var_mix], 1, A_cyc);
    lmiterm([-lmi_eig_mix, 1, 2, Y_var_mix], 1, C_cyc);
    lmiterm([-lmi_eig_mix, 2, 2, X_var_mix], 1, 1);
    
    % === LMI #3: X > eps*I ===
    lmi_pos_mix = newlmi;
    lmiterm([lmi_pos_mix, 1, 1, X_var_mix], -1, 1);
    lmiterm([lmi_pos_mix, 1, 1, 0], eps_val*eye(n_cyc));
    
    % === LMI #4: [W, I; I, X] >= 0 → W >= X^{-1} ===
    lmi_cov_mix = newlmi;
    lmiterm([-lmi_cov_mix, 1, 1, W_var_mix], 1, 1);
    lmiterm([-lmi_cov_mix, 1, 2, 0], eye(n_cyc));
    lmiterm([-lmi_cov_mix, 2, 2, X_var_mix], 1, 1);
    
    lmisys_mix = getlmis;
    
    % Objective: min trace(W)
    ndec_mix = decnbr(lmisys_mix);
    c_mix = zeros(ndec_mix, 1);
    for i = 1:ndec_mix
        ei = zeros(ndec_mix, 1);
        ei(i) = 1;
        Wi = dec2mat(lmisys_mix, ei, W_var_mix);
        c_mix(i) = trace(Wi);
    end
    
    [copt_mix, xopt_mix] = mincx(lmisys_mix, c_mix, options);
    
    if isempty(copt_mix)
        fprintf('Infeasible\n');
        results.r_bar(end+1) = r_bar;
        results.trace_W(end+1) = NaN;
        results.trace_Xinv(end+1) = NaN;
        results.max_eig(end+1) = NaN;
        results.feasible(end+1) = false;
        continue;
    end
    
    X_mix = dec2mat(lmisys_mix, xopt_mix, X_var_mix);
    Y_mix = dec2mat(lmisys_mix, xopt_mix, Y_var_mix);
    W_mix = dec2mat(lmisys_mix, xopt_mix, W_var_mix);
    
    trace_W_mix = trace(W_mix);
    trace_Xinv_mix = trace(inv(X_mix));
    
    % Gain recovery: L = -X^{-1}*Y
    L_mix = -X_mix \ Y_mix;
    
    A_cl_mix = A_cyc - L_mix * C_cyc;
    max_eig_mix = max(abs(eig(A_cl_mix)));
    
    fprintf('trace(W)=%.4f, trace(X^{-1})=%.4f, max|eig|=%.6f\n', ...
            trace_W_mix, trace_Xinv_mix, max_eig_mix);
    
    results.r_bar(end+1) = r_bar;
    results.trace_W(end+1) = trace_W_mix;
    results.trace_Xinv(end+1) = trace_Xinv_mix;
    results.max_eig(end+1) = max_eig_mix;
    results.feasible(end+1) = true;
end

%% 5. Results Summary
fprintf('\n=== Results Summary ===\n');
fprintf('r_bar\t\ttrace(W)\ttrace(X^{-1})\tmax|eig|\n');
fprintf('------------------------------------------------------------\n');
for i = 1:length(results.r_bar)
    if results.feasible(i)
        fprintf('%.3f\t\t%.4f\t\t%.4f\t\t%.6f\n', ...
                results.r_bar(i), results.trace_W(i), results.trace_Xinv(i), ...
                results.max_eig(i));
    else
        fprintf('%.3f\t\t-\t\t-\t\t-\n', results.r_bar(i));
    end
end

fprintf('\n=== Comparison ===\n');
fprintf('Optimal Kalman filter:  trace(W) = %.4f, max|eig| = %.6f\n', copt_kf, max(abs(eig_kf)));

feasible_idx = logical(results.feasible);
if any(feasible_idx)
    trace_W_vals = results.trace_W(feasible_idx);
    max_eig_vals = results.max_eig(feasible_idx);
    r_vals = results.r_bar(feasible_idx);
    
    fprintf('\nExpected behavior:\n');
    fprintf('  Large r_bar (relaxed) → trace(W) → Optimal Kalman value (%.4f)\n', copt_kf);
    fprintf('  Small r_bar (strict) → trace(W) increases\n');
    fprintf('  Always trace(W) >= %.4f (Optimal Kalman value)\n', copt_kf);
    
    if min(trace_W_vals) >= copt_kf - 0.01
        fprintf('\nVerification: OK - trace(W) >= Optimal Kalman value\n');
    else
        fprintf('\nVerification: NG - trace(W) < Optimal Kalman value (needs review)\n');
    end
end

%% 6. Plots
% Figure 1: trace(W) vs r_bar
figure('Position', [100 100 600 450]);
feasible_idx = logical(results.feasible);
if any(feasible_idx)
    plot(results.r_bar(feasible_idx), results.trace_W(feasible_idx), ...
         'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    yline(copt_kf, 'g--', 'LineWidth', 2);
    xline(max(abs(eig_kf)), 'g:', 'LineWidth', 1);
end
xlabel('r_{bar} (eigenvalue radius constraint)');
ylabel('trace(W)');
title('Kalman Filter / Eigenvalue Placement Trade-off');
legend({'Mixed design', 'Optimal Kalman', 'Optimal KF max|eig|'}, 'Location', 'best');
grid on;
set(gca, 'XDir', 'reverse');

% Figure 2: max|eig| vs r_bar
figure('Position', [750 100 600 450]);
if any(feasible_idx)
    plot(results.r_bar(feasible_idx), results.max_eig(feasible_idx), ...
         'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot(results.r_bar(feasible_idx), results.r_bar(feasible_idx), ...
         'k--', 'LineWidth', 1);
    yline(max(abs(eig_kf)), 'g:', 'LineWidth', 1);
end
xlabel('r_{bar} (eigenvalue radius constraint)');
ylabel('Actual max|eig|');
title('Eigenvalue Constraint Achievement');
legend({'Actual max|eig|', 'r_{bar}', 'Optimal KF max|eig|'}, 'Location', 'best');
grid on;
set(gca, 'XDir', 'reverse');

%% 7. Gain Comparison
fprintf('\n=== Gain Comparison ===\n');

extract_gains = @(L_cyc) extract_periodic_gains(L_cyc, n, m, N);

L_kf_cell = extract_gains(L_kf);

fprintf('Optimal Kalman filter L_0 (GPS + wheel speed):\n');
disp(L_kf_cell{1});

fprintf('Optimal Kalman filter L_1 (wheel speed only):\n');
disp(L_kf_cell{2});

% Strictest feasible solution gain
if any(feasible_idx)
    [~, min_idx] = min(results.r_bar(feasible_idx));
    feasible_r = results.r_bar(feasible_idx);
    r_strictest = feasible_r(min_idx);
    
    % Re-solve for that solution
    setlmis([]);
    [X_var_s, ~, ~] = lmivar(1, [n_cyc 1]);
    [Y_var_s, ~, ~] = lmivar(2, [n_cyc m_cyc]);
    [W_var_s, ~, ~] = lmivar(1, [n_cyc 1]);
    
    lmi_dari_s = newlmi;
    lmiterm([-lmi_dari_s, 1, 1, X_var_s], 1, 1);
    lmiterm([-lmi_dari_s, 1, 2, X_var_s], 1, A_cyc);
    lmiterm([-lmi_dari_s, 1, 2, Y_var_s], 1, C_cyc);
    lmiterm([-lmi_dari_s, 2, 2, X_var_s], 1, 1);
    lmiterm([-lmi_dari_s, 1, 3, X_var_s], 1, Q_cyc_sqrt);
    lmiterm([-lmi_dari_s, 3, 3, 0], eye(n_cyc));
    lmiterm([-lmi_dari_s, 1, 4, Y_var_s], 1, R_cyc_sqrt);
    lmiterm([-lmi_dari_s, 4, 4, 0], eye(m_cyc));
    
    lmi_eig_s = newlmi;
    lmiterm([-lmi_eig_s, 1, 1, X_var_s], r_strictest^2, 1);
    lmiterm([-lmi_eig_s, 1, 2, X_var_s], 1, A_cyc);
    lmiterm([-lmi_eig_s, 1, 2, Y_var_s], 1, C_cyc);
    lmiterm([-lmi_eig_s, 2, 2, X_var_s], 1, 1);
    
    lmi_pos_s = newlmi;
    lmiterm([lmi_pos_s, 1, 1, X_var_s], -1, 1);
    lmiterm([lmi_pos_s, 1, 1, 0], eps_val*eye(n_cyc));
    
    lmi_cov_s = newlmi;
    lmiterm([-lmi_cov_s, 1, 1, W_var_s], 1, 1);
    lmiterm([-lmi_cov_s, 1, 2, 0], eye(n_cyc));
    lmiterm([-lmi_cov_s, 2, 2, X_var_s], 1, 1);
    
    lmisys_s = getlmis;
    
    ndec_s = decnbr(lmisys_s);
    c_s = zeros(ndec_s, 1);
    for i = 1:ndec_s
        ei = zeros(ndec_s, 1);
        ei(i) = 1;
        Wi = dec2mat(lmisys_s, ei, W_var_s);
        c_s(i) = trace(Wi);
    end
    
    [~, xopt_s] = mincx(lmisys_s, c_s, options);
    
    if ~isempty(xopt_s)
        X_s = dec2mat(lmisys_s, xopt_s, X_var_s);
        Y_s = dec2mat(lmisys_s, xopt_s, Y_var_s);
        L_s = -X_s \ Y_s;
        
        L_s_cell = extract_gains(L_s);
        
        fprintf('\nStrictest constraint (r=%.3f) L_0:\n', r_strictest);
        disp(L_s_cell{1});
        
        fprintf('Strictest constraint (r=%.3f) L_1:\n', r_strictest);
        disp(L_s_cell{2});
    end
end

fprintf('\n=== Completed ===\n');

%% Helper Function
function L_cell = extract_periodic_gains(L_cyc, n, m, N)
    L_cell = cell(N, 1);
    for k = 1:N-1
        L_cell{k} = L_cyc(k*n+1:(k+1)*n, (k-1)*m+1:k*m);
    end
    L_cell{N} = L_cyc(1:n, (N-1)*m+1:N*m);
end