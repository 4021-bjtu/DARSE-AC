function k = darse_true_kernel(size_, kind)
%DARSE_TRUE_KERNEL Ground-truth kernel used only for synthesis/evaluation.
% Gaussian (isotropic) or linear-motion rotated by 23 degrees.
k = zeros(size_);
mid = ceil(size_ / 2);
if strcmp(kind, 'motion')
    k(mid, 2:size_ - 1) = 1;
    k = darse_rotate(k, 23);
elseif strcmp(kind, 'gaussian')
    k(mid, mid) = 1;
    k = darse_gauss_psf(size_, 1.4);
else
    error('unknown kernel kind');
end
k = darse_normalize_kernel(k);
end

function h = darse_gauss_psf(size_, sigma)
%DARSE_GAUSS_PSF Gaussian PSF of given size and standard deviation.
gx = exp(-0.5 * ((0:(size_ - 1)) - floor((size_ - 1) / 2)).^2 / sigma / sigma);
h = gx(:) * gx(:)';
h = h / sum(h(:));
end
