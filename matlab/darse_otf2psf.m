function k = darse_otf2psf(otf, shape)
%DARSE_OTF2PSF OTF -> centered spatial PSF (inverse of darse_psf2otf).
x = real(ifft2(otf));
x = circshift(x, [floor(shape(1)/2), floor(shape(2)/2)]);
k = x(1:shape(1), 1:shape(2));
end
