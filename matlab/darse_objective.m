function e = darse_objective(observed, image, kernel, z, cfg)
%DARSE_OBJECTIVE Diagnostic Eq.(5)-inspired energy per value, NOT a
% convergence certificate (mirror of Python objective()).
[gx, gy] = darse_grad(image);
[zx, zy] = darse_grad(z);
tmp = darse_convolve(image, kernel) - observed;
if cfg.curvature > 0
    curv = cfg.curvature .* abs(darse_curvature_filter(image));
else
    curv = zeros(size(image));
end
sparse_count = sum(sum((gx.^2 + gy.^2) > 1e-12));
e = sum(tmp(:).^2) ...
    + cfg.edge * (sum(sum((gx - zx).^2)) + sum(sum((gy - zy).^2))) ...
    + cfg.l0 * sparse_count ...
    + sum(curv(:));
end
