function img_o = imshift(img_i, s1, s2)
% =========================================================================
% Lateral translate the image with specified shift amount.
% -------------------------------------------------------------------------
% Input:    - img_i  : Input image.
%           - s1, s2 : Shifts along the two dimension (pixel).
% Output:   - img_o  : Shifted image.
% =========================================================================

% extract the size of the input image
[n2,n1] = size(img_i);

% frequency coordinate
f1 = -n1/2:1:n1/2-1;
f2 = -n2/2:1:n2/2-1;
[u1,u2] = meshgrid(f1,f2);

% FFT-based translation calculation
img_o = ifft2(fftshift(fftshift(fft2(img_i)).*exp(-1i*2*pi*(s1*u1/n1 + s2*u2/n2))));

end

