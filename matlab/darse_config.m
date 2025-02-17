function cfg = darse_config()
%DARSE_CONFIG Default configuration for the DARSE-AC MATLAB reference.
% Values mirror the Python reference in darse_ac.py and the paper Eq.(5)-(12).
cfg = struct();
cfg.kernel_size      = 15;   % odd, centered
cfg.levels           = 3;    % pyramid levels
cfg.outer            = 5;    % alternations per level
cfg.hqs_steps        = 6;    % inner HQS iterations
cfg.edge             = 4e-4; % lambda, gradient-image consistency
cfg.l0               = 4e-3; % xi, gradient L0 weight
cfg.curvature        = 4e-3; % theta, average curvature weight
cfg.ridge            = 2.0;  % tau, kernel ridge
cfg.guide            = 8e-4; % gamma, structure guidance strength
cfg.mu               = 0.01; % epsilon in mutual-guidance denominator
cfg.penalty_start    = 0.02; % initial HQS penalty
cfg.penalty_growth   = 2.0;  % penalty multiplier per step
cfg.curvature_steps  = 30;   % curvature prox steps
cfg.guide_steps      = 3;    % frozen-weight guidance steps
cfg.final_reg        = 0.002;% final non-blind regularization
end
