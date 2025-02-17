function y = darse_resize(x, target, mode)
%DARSE_RESIZE Pure-interpolation resize (nearest/bilinear), toolbox-free.
% Maps output pixel centers to input coordinates exactly like bilinear resizing:
%   in = (out + 0.5) * inSize/outSize - 0.5.
if nargin < 3
    mode = 'bilinear';
end
[h, w] = size(x);
H = target(1);
W = target(2);
y = zeros(H, W);
if strcmp(mode, 'nearest')
    for i = 1:H
        src_i = min(h, max(1, floor((i - 0.5) * h / H) + 1));
        for j = 1:W
            src_j = min(w, max(1, floor((j - 0.5) * w / W) + 1));
            y(i, j) = x(src_i, src_j);
        end
    end
else
    for i = 1:H
        si = (i - 0.5) * h / H - 0.5;
        i0 = max(1, floor(si) + 1);
        i1 = min(h, floor(si) + 2);
        ti = max(0, min(1, si - (i0 - 1)));
        if i0 == i1
            ti = 0;
        end
        for j = 1:W
            sj = (j - 0.5) * w / W - 0.5;
            j0 = max(1, floor(sj) + 1);
            j1 = min(w, floor(sj) + 2);
            tj = max(0, min(1, sj - (j0 - 1)));
            if j0 == j1
                tj = 0;
            end
            y(i, j) = (1 - ti) * ((1 - tj) * x(i0, j0) + tj * x(i0, j1)) ...
                    + ti * ((1 - tj) * x(i1, j0) + tj * x(i1, j1));
        end
    end
end
end
