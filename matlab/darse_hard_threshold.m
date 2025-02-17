function [bx, by] = darse_hard_threshold(gx, gy, weight, penalty)
%DARSE_HARD_THRESHOLD Joint vector L0 prox: keep pixel if ||grad||^2 > weight/penalty.
mask = (gx.^2 + gy.^2) > (weight / penalty);
bx = gx .* mask;
by = gy .* mask;
end
