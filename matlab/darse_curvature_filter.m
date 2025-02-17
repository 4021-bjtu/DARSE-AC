function y = darse_curvature_filter(x)
%DARSE_CURVATURE_FILTER Linear AC surrogate from paper Eq.(8): 3x3 stencil / 16.
stencil = [-1, 5, -1; 5, -16, 5; -1, 5, -1] / 16;
y = darse_convolve(x, stencil);
end
