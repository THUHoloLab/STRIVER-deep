function img_o = padimage(img_i, bias, size_out)
% =========================================================================
% Zero-pad an 2D image.
% -------------------------------------------------------------------------
% Input:    - img_i    : Input image (2D array).
%           - bias     : Padding sizes before the image along the two axes.
%           - size_out : Output image size.
% Output:   - img_o    : Output zero-padded image.
% =========================================================================

[n1,n2] = size(img_i);

img_o = zeros(size_out);

img_o(bias(1)+1:bias(1)+n1,bias(2)+1:bias(2)+n2) = img_i;

end