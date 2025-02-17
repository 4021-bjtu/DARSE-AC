function r = darse_rotate(x, degrees)
%DARSE_ROTATE Free 2D rotation with bilinear sampling and same-size output
% about the array center. Toolbox-free replacement for imrotate(...,'bilinear','crop').
[h, w] = size(x);
r = zeros(h, w);
theta = degrees * pi / 180;
ct = cos(theta);
st = sin(theta);
cx = (w + 1) / 2;
cy = (h + 1) / 2;
for i = 1:h
    y = i - cy;
    for j = 1:w
        xin = j - cx;
        src_x = ct * xin + st * y + cx;
        src_y = -st * xin + ct * y + cy;
        if src_x >= 1 && src_x <= w && src_y >= 1 && src_y <= h
            x0 = max(1, floor(src_x));
            x1 = min(w, x0 + 1);
            y0 = max(1, floor(src_y));
            y1 = min(h, y0 + 1);
            tx = src_x - x0;
            ty = src_y - y0;
            v00 = x(y0, x0); v01 = x(y0, x1);
            v10 = x(y1, x0); v11 = x(y1, x1);
            r(i, j) = (1 - ty) * ((1 - tx) * v00 + tx * v01) ...
                    + ty * ((1 - tx) * v10 + tx * v11);
        end
    end
end
end
