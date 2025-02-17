function d = darse_adjoint(gx, gy)
%DARSE_ADJOINT Exact adjoint D* of darse_grad (negative backward divergence).
d = (circshift(gx, [0, 1]) - gx) + (circshift(gy, [1, 0]) - gy);
end
