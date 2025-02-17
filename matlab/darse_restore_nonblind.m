function x = darse_restore_nonblind(observed, kernel, regularization)
%DARSE_RESTORE_NONBLIND Final gradient-Tikhonov (Wiener-like) solve.
% Substitute for the paper's final non-blind step, same functional form as the
% Python reference restore_nonblind.
K = darse_psf2otf(kernel, size(observed));
denom = abs(K).^2 + regularization .* darse_gradient_power(size(observed));
x = real(ifft2(conj(K) .* fft2(observed) ./ max(denom, 1e-12)));
x = min(max(x, 0), 1);
end
