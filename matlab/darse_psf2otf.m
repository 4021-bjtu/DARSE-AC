function otf = darse_psf2otf(kernel, shape)
%DARSE_PSF2OTF Centered spatial PSF -> OTF with explicit ceil-based shift,
% matching the Python reference (roll by -(size/2) on each axis).
kh = size(kernel, 1);
kw = size(kernel, 2);
padded = zeros(shape);
padded(1:kh, 1:kw) = kernel;
padded = circshift(padded, -[floor(kh/2), floor(kw/2)]);
otf = fft2(padded);
end
