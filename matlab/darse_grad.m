function [gx, gy] = darse_grad(x)
%DARSE_GRAD Periodic forward differences (horizontal then vertical).
% Same convention as the Python reference: gx = roll(-1, axis=1) - x.
gx = circshift(x, [0, -1]) - x;
gy = circshift(x, [-1, 0]) - x;
end
