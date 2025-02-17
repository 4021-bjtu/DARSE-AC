function d = darse_adjointMatrix(wx, wy, v)
%DARSE_ADJOINTMATRIX D' * (W .* D) v for weighted periodic gradients.
% Solves the weighted Laplacian structure needed by the guidance filter.
[dgx, dgy] = darse_grad(v);
d = darse_adjoint(wx .* dgx, wy .* dgy);
end
