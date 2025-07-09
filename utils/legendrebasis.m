function [A, xx, yy, mask_nan] = legendrebasis(img, max_ord, n_rect, mask_nan)
% =========================================================================
% Select the background region and prepare the Legendre polynomials.
% -------------------------------------------------------------------------
% Input:    - img      : Input image.
%           - max_ord  : Maximum polynomial order.
%           - n_rect   : Number of crop rectangles to select the background
%                        region.
%           - mask_nan : Mask for selecting the background (default: []).
% Output:   - A        : Matrix consisting of Legendre polynomials.
%           - xx / yy  : 2D coordinates.
%           - mask_nan : Updated mask. 
% =========================================================================

n1 = size(img,1);
n2 = size(img,2);

c1 = linspace(0,1,n1);
c2 = linspace(0,1,n2);

[xx,yy] = meshgrid(c2,c1);

img_tmp = img;

% select crop rectangles
mask = zeros(n1,n2);
if isempty(mask_nan)
    for ord = 1:n_rect
        fig = figure;
        tmp = (img_tmp - min(img_tmp(:)))/(max(img_tmp(:)) - min(img_tmp(:)));
        [~,rect_tmp] = imcrop(tmp);
        close(fig);
        mask(round(rect_tmp(2)):round(rect_tmp(2)+rect_tmp(4)), ...
            round(rect_tmp(1)):round(rect_tmp(1)+rect_tmp(3))) = 1;
        img_tmp(round(rect_tmp(2)):round(rect_tmp(2)+rect_tmp(4)), ...
            round(rect_tmp(1)):round(rect_tmp(1)+rect_tmp(3))) = min(img_tmp(:));
    end
    mask_nan = nan(n1,n2);
    mask_nan(mask == 1) = 1;
end

xxx = mask_nan .* xx;
yyy = mask_nan .* yy;

xxx = xxx(:);
yyy = yyy(:);

xxx(isnan(xxx)) = [];
yyy(isnan(yyy)) = [];

% calculate the polynomials
n_coef = (max_ord+1)*(max_ord+2)/2;
A = nan(length(xxx),n_coef);
index = 1;
for ord = 0:max_ord
    for i = 0:ord
        j = ord-i;
        A(:,index) = myLegendreP(i,xxx).*myLegendreP(j,yyy);
        index = index + 1;
    end
end

end

