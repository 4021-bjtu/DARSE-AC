function p = darse_gradient_power(shape)
%DARSE_GRADIENT_POWER |D|^2 frequency weights for the image update.
h = shape(1);
w = shape(2);
fx = exp(2i * pi * ((0:(w-1)) / w)) - 1;
fy = exp(2i * pi * ((0:(h-1)) / h)) - 1;
px = abs(fx).^2;
py = abs(fy).^2;
p = px(ones(h, 1), :);
p = p + py(:, ones(w, 1));
end
