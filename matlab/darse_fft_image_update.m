function x = darse_fft_image_update(observed, kernel, z, alpha, beta, edge, penalty, alpha_penalty)
%DARSE_FFT_IMAGE_UPDATE Exact unconstrained quadratic minimizer of the image
% block of the HQS-split objective (mirror of Python fft_image_update).
K = darse_psf2otf(kernel, size(observed));
power = darse_gradient_power(size(observed));
rhs = conj(K) .* fft2(observed);
rhs = rhs + edge * power .* fft2(z);
rhs = rhs + penalty * fft2(darse_adjoint(beta{1}, beta{2}));
rhs = rhs + alpha_penalty * fft2(alpha);
denominator = abs(K).^2 + (edge + penalty) * power + alpha_penalty;
x = real(ifft2(rhs ./ max(denominator, 1e-15)));
end
