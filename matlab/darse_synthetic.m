function img = darse_synthetic(size_)
%DARSE_SYNTHETIC Generated text-and-geometry example (NOT a cultural artifact).
% Returns a uint8-nd float image in [0, 1]. Mirrors experiment.synthetic_image.
canvas = ones(size_ * 2, size_ * 2) * 0.92;
% Draw via a simple stroke painter on a low-res bitmap then smooth.
bits = canvas;
font_h = round(size_ / 5);
% Text rows: row bands with a few vertical strokes to fake letter strokes.
band1 = round(size_ * 0.12):round(size_ * 0.12) + font_h;
bit = zeros(size_ * 2, 1);
n_strokes = 7;
for s = 1:n_strokes
    x0 = round(size_ * 0.12 + (s - 1) * size_ * 0.10);
    bits(band1, x0:min(x0 + round(size_ * 0.02), size(bits, 2))) = 0.12;
end
band2 = round(size_ * 0.48):round(size_ * 0.48) + round(size_ / 8);
for s = 1:5
    x0 = round(size_ * 0.15 + (s - 1) * size_ * 0.13);
    bits(band2, x0:min(x0 + round(size_ * 0.015), size(bits, 2))) = 0.2;
end
% Rectangle
r8 = round(size_ : size_ * 1.75);
c8 = round(size_ * 0.16 : size_ * 0.88);
bits(r8(1):r8(end), c8([1, end])) = 0.15;
bits(r8([1, end]), c8(1):c8(end)) = 0.15;
% Darken interior strokes of rectangle
for i = 1:5
    y = round(size_ * (1.1 + (i - 1) * 0.12));
    bits(y:y + 1, round(size_ * 0.25):round(size_ * 0.75)) = 0.15;
end
% Circle (ring) drawn with a contour mask at radius
cx = size_ * 1.45; cy = size_ * 1.38;
[Y, X] = ndgrid(1:size_ * 2, 1:size_ * 2);
radius = sqrt((X - cx).^2 + (Y - cy).^2);
ring = abs(radius - size_ * 0.33) < 2.2;
bits(ring) = 0.3;
img = darse_resize(bits, [size_, size_], 'bilinear');
img = min(max(img, 0), 1);
end
