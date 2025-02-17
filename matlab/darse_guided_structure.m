function z = darse_guided_structure(image, strength, mu, steps)
%DARSE_GUIDED_STRUCTURE Frozen-weight mutual-guidance surrogate (Eq.12, mu=0.01).
% Each iteration freezes w = 1./max(abs(Dz), mu) and solves
%   (I + strength * D' w D) z = image  via a bundled CG. Uses explicit 2D
% indexing inside matvec so vectorized size mismatches cannot occur.
if strength == 0
    z = image;
    return;
end
z = image;
for s = 1:steps
    [gx, gy] = darse_grad(z);
    wx = 1 ./ max(abs(gx), mu);
    wy = 1 ./ max(abs(gy), mu);
    [h, w] = size(image);
    matvec = @(v) darse_matvec2d(wx, wy, v, h, w, strength);
    z = darse_pcg(matvec, image(:), z(:), 1e-6, 150);
    if any(~isfinite(z))
        z = image;
        break;
    end
    z = reshape(z, size(image));
end
end
