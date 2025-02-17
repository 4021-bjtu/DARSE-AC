function y = darse_matvec2d(wx, wy, v, h, w, strength)
%DARSE_MATVEC2D y = v + strength * D' (W .* D) v, all arrays forced to h-by-w.
v2 = reshape(v, h, w);
[dgx, dgy] = darse_grad(v2);
    y = reshape(v2 + strength * darse_adjoint(wx .* dgx, wy .* dgy), h * w, 1);
end
