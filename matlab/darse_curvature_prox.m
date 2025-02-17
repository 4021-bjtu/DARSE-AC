function alpha = darse_curvature_prox(image, strength, steps)
%DARSE_CURVATURE_PROX Dual projected-gradient prox of
%   .5||alpha-image||^2 + strength*||C*alpha||_1
% via alpha = image - C* p with |p| <= strength; C symmetric.
% Mirrors the Python reference; safeguards the surrogate energy.
if strength == 0
    alpha = image;
    return;
end
C = darse_psf2otf([-1, 5, -1; 5, -16, 5; -1, 5, -1] / 16, size(image));
inv_norm = 1 / (max(abs(C(:))).^2);
dual = zeros(size(image));
alpha = image;
for it = 1:steps
    ca = real(ifft2(C .* fft2(alpha)));
    dual = min(max(dual + ca * inv_norm, -strength), strength);
    alpha = image - real(ifft2(conj(C) .* fft2(dual)));
end
initial = strength * sum(abs(darse_curvature_filter(image)(:)));
energy = 0.5 * sum(sum((alpha - image).^2)) + strength * sum(abs(darse_curvature_filter(alpha)(:)));
if energy > initial + 1e-10
    alpha = image;
end
end
