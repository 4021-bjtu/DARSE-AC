function s = darse_ssim(a, b)
%DARSE_SSIM Simple structural similarity (data range 1).
% For the demo only; skimage structural_similarity uses slightly different
% constants, so compare trends, not absolute equality, across implementations.
k1 = 0.01; k2 = 0.03;
C1 = (k1 * 1)^2; C2 = (k2 * 1)^2;
w = darse_gaussian_window(11, 1.5);
mu_a = darse_filter_window(a, w);
mu_b = darse_filter_window(b, w);
mu_a2 = mu_a .^ 2; mu_b2 = mu_b .^ 2; mu_ab = mu_a .* mu_b;
sig_a2 = darse_filter_window(a .^ 2, w) - mu_a2;
sig_b2 = darse_filter_window(b .^ 2, w) - mu_b2;
sig_ab = darse_filter_window(a .* b, w) - mu_ab;
numerator = (2 * mu_ab + C1) .* (2 * sig_ab + C2);
denominator = (mu_a2 + mu_b2 + C1) .* (sig_a2 + sig_b2 + C2);
s = mean(mean(numerator ./ max(denominator, 1e-12)));
end

function w = darse_gaussian_window(n, sigma)
%DARSE_GAUSSIAN_WINDOW Separable Gaussian kernel of size n x n.
x = ((1:n) - (n + 1) / 2).^2;
g = exp(-x / (2 * sigma^2));
g = g / sum(g);
w = g(:) * g(:)';
end

function y = darse_filter_window(x, w)
%DARSE_FILTER_WINDOW Border-symmetric correlation with w (imfilter 'symmetric').
[h, k] = size(w);
r = floor(k / 2); c = floor(h / 2);
xr = [flipud(x(2:r+1, :)); x; flipud(x(end-r:end-1, :))];
xc = [fliplr(xr(:, 2:c+1)), xr, fliplr(xr(:, end-c:end-1))];
y = conv2(xc, rot90(w, 2), 'valid');
end
