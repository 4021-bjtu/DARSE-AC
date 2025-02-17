function y = darse_convolve(x, k)
%DARSE_CONVOLVE Periodic convolution of x with centered kernel k via FFT.
y = real(ifft2(fft2(x) .* darse_psf2otf(k, size(x))));
end
