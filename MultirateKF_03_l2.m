%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multirate Kalman Filter - Kalman Filter/l2-induced Norm Mixed Design
%
% Author: Hiroshi Okajima
% Created with assistance from Claude (Anthropic)
% Date: February 2025
%
% Description:
%   This code implements mixed design combining optimal Kalman filtering
%   and l2-induced norm constraints for multirate systems.
%   The unified LMI framework handles both objectives simultaneously:
%   - Optimal Kalman filter: Minimizes estimation error covariance
%   - l2-induced norm: Limits worst-case disturbance amplification
%
% Unified LMI Structure:
%   [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
%   [*,    (2,2),   0,         0       ]
%   [*,    0,       (3,3),     0       ]
%   [*,    0,       0,         (4,4)   ] > 0
%
%   DARI (Optimal Kalman): (2,2)=X,   (3,3)=I,    (4,4)=I
%   SICE (l2):            (2,2)=X-I, (3,3)=γ²I,  (4,4)=γ²I
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
%   2. Compute optimal l2-induced norm design
%   3. Sweep l2 constraint levels (γ_bar from 10×γ_opt to 1.01×γ_opt)
%   4. Analyze trade-off between average and worst-case performance
%   5. Compare gains for different designs
%
% Output:
%   - Console: Trade-off results, gain comparisons
%   - Figures: trace(W) vs γ_bar/γ_opt, max|eig| vs γ_bar/γ_opt
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
fprintf('Kalman Filter/l2-induced Norm Mixed Design (Unified Form)\n');
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

%% 3. Optimal Kalman Filter (DARI only)
fprintf('=== Optimal Kalman Filter (DARI only) ===\n');
fprintf('LMI: (2,2)=X, (3,3)=I, (4,4)=I\n\n');

setlmis([]);

[X_var_kf, ~, ~] = lmivar(1, [n_cyc 1]);
[Y_var_kf, ~, ~] = lmivar(2, [n_cyc m_cyc]);
[W_var_kf, ~, ~] = lmivar(1, [n_cyc 1]);

% LMI #1: DARI
% [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
% [*,    X,       0,         0       ]
% [*,    0,       I,         0       ]
% [*,    0,       0,         I       ] > 0
lmi_dari_kf = newlmi;
lmiterm([-lmi_dari_kf, 1, 1, X_var_kf], 1, 1);
lmiterm([-lmi_dari_kf, 1, 2, X_var_kf], 1, A_cyc);
lmiterm([-lmi_dari_kf, 1, 2, Y_var_kf], 1, C_cyc);
lmiterm([-lmi_dari_kf, 2, 2, X_var_kf], 1, 1);
lmiterm([-lmi_dari_kf, 1, 3, X_var_kf], 1, Q_cyc_sqrt);
lmiterm([-lmi_dari_kf, 3, 3, 0], eye(n_cyc));
lmiterm([-lmi_dari_kf, 1, 4, Y_var_kf], 1, R_cyc_sqrt);
lmiterm([-lmi_dari_kf, 4, 4, 0], eye(m_cyc));

% LMI #2: X > eps*I
lmi_pos_kf = newlmi;
lmiterm([lmi_pos_kf, 1, 1, X_var_kf], -1, 1);
lmiterm([lmi_pos_kf, 1, 1, 0], eps_val*eye(n_cyc));

% LMI #3: [W, I; I, X] >= 0  → W >= X^{-1}
lmi_cov_kf = newlmi;
lmiterm([-lmi_cov_kf, 1, 1, W_var_kf], 1, 1);
lmiterm([-lmi_cov_kf, 1, 2, 0], eye(n_cyc));
lmiterm([-lmi_cov_kf, 2, 2, X_var_kf], 1, 1);

lmisys_kf = getlmis;

% Objective: min trace(W)
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

% Gain recovery: L = -X^{-1}*Y
L_kf = -X_kf \ Y_kf;

A_cl_kf = A_cyc - L_kf * C_cyc;
eig_kf = eig(A_cl_kf);

fprintf('Optimal Kalman filter solution:\n');
fprintf('  trace(W) = %.4f\n', copt_kf);
fprintf('  trace(X^{-1}) = %.4f\n', trace(inv(X_kf)));
fprintf('  max|eig| = %.6f\n\n', max(abs(eig_kf)));

%% 4. l2-induced Norm Optimal (SICE Eq.(19) only)
fprintf('=== l2 Optimal (SICE Eq.(19) only) ===\n');
fprintf('LMI: (2,2)=X-I, (3,3)=γ²I, (4,4)=γ²I\n\n');

setlmis([]);

[gamma2_var_l2, ~, ~] = lmivar(1, [1 1]);
[X_var_l2, ~, ~] = lmivar(1, [n_cyc 1]);
[Y_var_l2, ~, ~] = lmivar(2, [n_cyc m_cyc]);

% LMI: SICE Eq.(19)
% [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
% [*,    X-I,     0,         0       ]
% [*,    0,       γ²I,       0       ]
% [*,    0,       0,         γ²I     ] > 0
lmi_sice_l2 = newlmi;
lmiterm([-lmi_sice_l2, 1, 1, X_var_l2], 1, 1);
lmiterm([-lmi_sice_l2, 1, 2, X_var_l2], 1, A_cyc);
lmiterm([-lmi_sice_l2, 1, 2, Y_var_l2], 1, C_cyc);
lmiterm([-lmi_sice_l2, 2, 2, X_var_l2], 1, 1);
lmiterm([-lmi_sice_l2, 2, 2, 0], -0.1*eye(n_cyc));  % X - I
lmiterm([-lmi_sice_l2, 1, 3, X_var_l2], 1, Q_cyc_sqrt);
lmiterm([-lmi_sice_l2, 3, 3, gamma2_var_l2], 1, eye(n_cyc));  % γ²I
lmiterm([-lmi_sice_l2, 1, 4, Y_var_l2], 1, R_cyc_sqrt);
lmiterm([-lmi_sice_l2, 4, 4, gamma2_var_l2], 1, eye(m_cyc));  % γ²I

% X > eps*I
lmi_pos_l2 = newlmi;
lmiterm([lmi_pos_l2, 1, 1, X_var_l2], -1, 1);
lmiterm([lmi_pos_l2, 1, 1, 0], eps_val*eye(n_cyc));

lmisys_l2 = getlmis;

% Objective: min γ²
ndec_l2 = decnbr(lmisys_l2);
c_l2 = zeros(ndec_l2, 1);
for i = 1:ndec_l2
    ei = zeros(ndec_l2, 1);
    ei(i) = 1;
    g2i = dec2mat(lmisys_l2, ei, gamma2_var_l2);
    c_l2(i) = trace(g2i);
end

[copt_l2, xopt_l2] = mincx(lmisys_l2, c_l2, options);

if isempty(copt_l2)
    error('l2 optimization is infeasible');
end

gamma_l2_opt = sqrt(copt_l2);
X_l2 = dec2mat(lmisys_l2, xopt_l2, X_var_l2);
Y_l2 = dec2mat(lmisys_l2, xopt_l2, Y_var_l2);
L_l2 = -X_l2 \ Y_l2;

A_cl_l2 = A_cyc - L_l2 * C_cyc;
eig_l2 = eig(A_cl_l2);

fprintf('l2 optimal solution:\n');
fprintf('  Optimal γ = %.6f\n', gamma_l2_opt);
fprintf('  trace(X^{-1}) = %.4f\n', trace(inv(X_l2)));
fprintf('  max|eig| = %.6f\n\n', max(abs(eig_l2)));

%% 5. Kalman Filter/l2 Mixed Design (DARI + SICE Eq.(19))
fprintf('=== Kalman Filter/l2 Mixed Design ===\n');
fprintf('LMI #1 (DARI):  (2,2)=X,   (3,3)=I,      (4,4)=I\n');
fprintf('LMI #2 (SICE):  (2,2)=X-I, (3,3)=γ_bar²I, (4,4)=γ_bar²I\n');
fprintf('Common variables: X, Y\n');
fprintf('Objective: min trace(W) where W >= X^{-1}\n\n');

gamma_bar_list = [gamma_l2_opt*10, gamma_l2_opt*5, gamma_l2_opt*3, gamma_l2_opt*2, ...
                  gamma_l2_opt*1.5, gamma_l2_opt*1.3, gamma_l2_opt*1.2, ...
                  gamma_l2_opt*1.1, gamma_l2_opt*1.05, gamma_l2_opt*1.02, gamma_l2_opt*1.01];
gamma_bar_list = sort(gamma_bar_list, 'descend');

results = struct();
results.gamma_bar = [];
results.trace_W = [];
results.trace_Xinv = [];
results.max_eig = [];
results.feasible = [];

for idx = 1:length(gamma_bar_list)
    gamma_bar = gamma_bar_list(idx);
    
    fprintf('γ_bar = %.4f (%.2f×γ_opt): ', gamma_bar, gamma_bar/gamma_l2_opt);
    
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
    
    % === LMI #2: SICE Eq.(19) (l2-induced norm constraint) ===
    % [X,    XA+YC,   XQ^{1/2},  YR^{1/2}]
    % [*,    X-I,     0,         0       ]
    % [*,    0,       γ²I,       0       ]
    % [*,    0,       0,         γ²I     ] > 0
    lmi_sice_mix = newlmi;
    lmiterm([-lmi_sice_mix, 1, 1, X_var_mix], 1, 1);
    lmiterm([-lmi_sice_mix, 1, 2, X_var_mix], 1, A_cyc);
    lmiterm([-lmi_sice_mix, 1, 2, Y_var_mix], 1, C_cyc);
    lmiterm([-lmi_sice_mix, 2, 2, X_var_mix], 1, 1);
    lmiterm([-lmi_sice_mix, 2, 2, 0], -0.1*eye(n_cyc));  % X - I
    lmiterm([-lmi_sice_mix, 1, 3, X_var_mix], 1, Q_cyc_sqrt);
    lmiterm([-lmi_sice_mix, 3, 3, 0], gamma_bar^2 * eye(n_cyc));  % γ_bar²I
    lmiterm([-lmi_sice_mix, 1, 4, Y_var_mix], 1, R_cyc_sqrt);
    lmiterm([-lmi_sice_mix, 4, 4, 0], gamma_bar^2 * eye(m_cyc));  % γ_bar²I
    
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
        results.gamma_bar(end+1) = gamma_bar;
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
    
    results.gamma_bar(end+1) = gamma_bar;
    results.trace_W(end+1) = trace_W_mix;
    results.trace_Xinv(end+1) = trace_Xinv_mix;
    results.max_eig(end+1) = max_eig_mix;
    results.feasible(end+1) = true;
end

%% 6. Results Summary
fprintf('\n=== Results Summary ===\n');
fprintf('γ_bar\t\ttrace(W)\ttrace(X^{-1})\tmax|eig|\tγ/γ_opt\n');
fprintf('------------------------------------------------------------------------\n');
for i = 1:length(results.gamma_bar)
    if results.feasible(i)
        gamma_ratio = results.gamma_bar(i) / gamma_l2_opt;
        fprintf('%.4f\t\t%.4f\t\t%.4f\t\t%.6f\t%.2f\n', ...
                results.gamma_bar(i), results.trace_W(i), results.trace_Xinv(i), ...
                results.max_eig(i), gamma_ratio);
    else
        fprintf('%.4f\t\t-\t\t-\t\t-\t\t%.2f\n', ...
                results.gamma_bar(i), results.gamma_bar(i)/gamma_l2_opt);
    end
end

fprintf('\n=== Comparison ===\n');
fprintf('Optimal Kalman filter: trace(W) = %.4f, trace(X^{-1}) = %.4f\n', copt_kf, trace(inv(X_kf)));
fprintf('l2 optimal: γ = %.4f, trace(X^{-1}) = %.4f\n', gamma_l2_opt, trace(inv(X_l2)));

feasible_idx = logical(results.feasible);
if any(feasible_idx)
    trace_W_vals = results.trace_W(feasible_idx);
    gamma_vals = results.gamma_bar(feasible_idx);
    
    fprintf('\nMixed design:\n');
    fprintf('  γ_bar = %.4f (%.1f×γ_opt): trace(W) = %.4f\n', ...
            gamma_vals(1), gamma_vals(1)/gamma_l2_opt, trace_W_vals(1));
    if length(gamma_vals) > 1
        fprintf('  γ_bar = %.4f (%.2f×γ_opt): trace(W) = %.4f\n', ...
                gamma_vals(end), gamma_vals(end)/gamma_l2_opt, trace_W_vals(end));
    end
    
    fprintf('\nExpected behavior:\n');
    fprintf('  γ_bar → ∞: trace(W) → Optimal Kalman value (%.4f)\n', copt_kf);
    fprintf('  γ_bar → γ_opt: l2 constraint tightens, infeasible or trace(W) increases\n');
    fprintf('  Always trace(W) >= %.4f (Optimal Kalman value)\n', copt_kf);
    
    if min(trace_W_vals) >= copt_kf - 0.01
        fprintf('\nVerification: OK - Mixed design trace(W) >= Optimal Kalman\n');
    else
        fprintf('\nVerification: NG - Mixed design trace(W) < Optimal Kalman (needs review)\n');
    end
end

%% 7. Plots
% Figure 1: trace(W) vs gamma_bar
figure('Position', [100 100 600 450]);
feasible_idx = logical(results.feasible);
if any(feasible_idx)
    semilogx(results.gamma_bar(feasible_idx)/gamma_l2_opt, results.trace_W(feasible_idx), ...
             'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    yline(copt_kf, 'g--', 'LineWidth', 2);
    yline(trace(inv(X_l2)), 'r--', 'LineWidth', 2);
    xline(1, 'k:', 'LineWidth', 1);
end
xlabel('\gamma_{bar} / \gamma_{opt}');
ylabel('trace(W)');
title('Kalman Filter / l_2 Trade-off');
legend({'Mixed design', 'Optimal Kalman', 'l_2 optimal', '\gamma_{opt}'}, 'Location', 'best');
grid on;

% Figure 2: max|eig| vs gamma_bar
figure('Position', [750 100 600 450]);
if any(feasible_idx)
    semilogx(results.gamma_bar(feasible_idx)/gamma_l2_opt, results.max_eig(feasible_idx), ...
             'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    yline(1, 'k--', 'LineWidth', 1);
    yline(max(abs(eig_kf)), 'g:', 'LineWidth', 1);
    yline(max(abs(eig_l2)), 'r:', 'LineWidth', 1);
end
xlabel('\gamma_{bar} / \gamma_{opt}');
ylabel('max|\lambda|');
title('Stability Margin');
legend({'Mixed design', 'Stability limit', 'Optimal Kalman', 'l_2 optimal'}, 'Location', 'best');
grid on;

%% 8. Gain Display
fprintf('\n=== Gain Comparison ===\n');

extract_gains = @(L_cyc) extract_periodic_gains(L_cyc, n, m, N);

L_kf_cell = extract_gains(L_kf);
L_l2_cell = extract_gains(L_l2);

fprintf('L_0 (GPS + wheel speed):\n');
fprintf('  Optimal Kalman:\n');
disp(L_kf_cell{1});
fprintf('  l2:\n');
disp(L_l2_cell{1});

fprintf('L_1 (wheel speed only):\n');
fprintf('  Optimal Kalman:\n');
disp(L_kf_cell{2});
fprintf('  l2:\n');
disp(L_l2_cell{2});

fprintf('\n=== Completed ===\n');

%% Helper Function
function L_cell = extract_periodic_gains(L_cyc, n, m, N)
    L_cell = cell(N, 1);
    for k = 1:N-1
        L_cell{k} = L_cyc(k*n+1:(k+1)*n, (k-1)*m+1:k*m);
    end
    L_cell{N} = L_cyc(1:n, (N-1)*m+1:N*m);
end