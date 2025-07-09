function imgfit = legendrepolyfit(xx, yy, max_ord, coefs)
% =========================================================================
% Calculate the linear combination of 2D Legendre polynomials given the
% specified coefficients.
% -------------------------------------------------------------------------
% Input:    - xx / yy : Input 2D coordinates.
%           - max_ord : Maximum order.
%           - coefs   : Coefficients.
% Output:   - imgfit  : Fitted results.
% =========================================================================

index = 1;
imgfit = zeros(size(xx));
for ord = 0:max_ord
    for i = 0:ord
        j = ord-i;
        imgfit = imgfit + coefs(index) * myLegendreP(i,xx).*myLegendreP(j,yy);
        index = index + 1;
    end
end

end