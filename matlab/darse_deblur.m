function [final, kernel, trace] = darse_deblur(observed, cfg)
%DARSE_DEBLUR Blind multi-scale reconstruction (mirror of Python darse_ac.deblur).
% Returns [final_image, kernel, trace]. No ground truth is used inside.
% Integer sizes are handled with round(); the reference uses scale factors
% 1/2, then 1, omitting scales whose min dimension falls below 24 px.
% Periodic boundaries and centered odd kernels are assumed throughout.

if nargin < 2
    cfg = darse_config();
end
if cfg.kernel_size >= min(size(observed))
    error('kernel_size must be smaller than the image');
end
scales = 0.5 .^ (cfg.levels - 1:-1:0);
keep = scales == 1 | (min(size(observed)) * scales >= 24);
scales = scales(keep);
kernel = [];
image = [];
trace = cell(1, numel(scales) * cfg.outer);
idx = 0;
for lv = 1:numel(scales)
    s = scales(lv);
    shape = max(8, round(size(observed) * s));
    a = darse_resize(double(observed), shape, 'bilinear');
    ker_size = max(3, round(cfg.kernel_size * s));
    if mod(ker_size, 2) == 0
        ker_size = ker_size + 1;
    end
    ker_size = min(ker_size, cfg.kernel_size);
    if isempty(kernel)
        kernel = zeros(ker_size, ker_size);
        kernel(ceil(ker_size/2), ceil(ker_size/2)) = 1;
        image = a;
    else
        kernel = darse_normalize_kernel(darse_resize(kernel, [ker_size, ker_size], 'bilinear'));
        image = darse_resize(image, shape, 'bilinear');
    end
    for it = 1:cfg.outer
        z = darse_guided_structure(image, cfg.guide, cfg.mu, cfg.guide_steps);
        before = image;
        for j = 1:cfg.hqs_steps
            penalty = cfg.penalty_start * cfg.penalty_growth ^ (j - 1);
            if cfg.curvature > 0
                alpha_penalty = penalty;
            else
                alpha_penalty = 0;
            end
            alpha = darse_curvature_prox(image, cfg.curvature / (2 * penalty), cfg.curvature_steps);
            [igx, igy] = darse_grad(image);
            [bx, by] = darse_hard_threshold(igx, igy, cfg.l0, penalty);
            image = darse_fft_image_update(a, kernel, z, alpha, {bx, by}, cfg.edge, penalty, alpha_penalty);
        end
        z = darse_guided_structure(image, cfg.guide, cfg.mu, cfg.guide_steps);
        kernel = darse_estimate_kernel(a, z, ker_size, cfg.ridge);
        if cfg.curvature > 0
            curv = cfg.curvature .* abs(darse_curvature_filter(image));
        else
            curv = zeros(size(image));
        end
        idx = idx + 1;
        trace{idx} = struct('level', lv - 1, 'scale', s, 'iteration', it - 1, ...
            'height', shape(1), 'width', shape(2), 'kernel_size', ker_size, ...
            'energy_per_pixel', darse_objective(a, image, kernel, z, cfg) / numel(a), ...
            'relative_change', norm(image(:) - before(:)) / max(norm(before(:)), 1e-12), ...
            'kernel_sum', sum(kernel(:)));
    end
end
final = darse_restore_nonblind(double(observed), kernel, cfg.final_reg);
end
