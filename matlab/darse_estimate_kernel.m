function k = darse_estimate_kernel(observed, z, size_out, ridge)
%DARSE_ESTIMATE_KERNEL Gradient-domain Fourier ridge solution followed by
% support projection (mirror of Python estimate_kernel).
[zx, zy] = darse_grad(z);
[ax, ay] = darse_grad(observed);
Zx = fft2(zx);
Zy = fft2(zy);
Ax = fft2(ax);
Ay = fft2(ay);
K = (conj(Zx) .* Ax + conj(Zy) .* Ay) ./ (abs(Zx).^2 + abs(Zy).^2 + ridge);
k = darse_normalize_kernel(darse_otf2psf(K, [size_out, size_out]));
end
