function k = darse_normalize_kernel(k)
%DARSE_NORMALIZE_KERNEL Nonnegative projection and unit sum, with impulse fallback.
k = max(k, 0);
s = sum(k(:));
if s <= 1e-15
    k = zeros(size(k));
    k(ceil(end/2), ceil(end/2)) = 1;
else
    k = k / s;
end
end
