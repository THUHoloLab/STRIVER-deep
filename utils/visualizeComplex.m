function [img,cbarimg] = visualizeComplex(cimg, cmap, amp_range, mode, reverse)
% =========================================================================
% Visualize complex-valued images via color coding. The amplitude and phase
% are represented by lightness / value and hue, respectively.
% -------------------------------------------------------------------------
% Input:    - cimg      : Input complex-valued image.
%           - cmap      : Colormap for phase visualization (cyclic
%                         colormaps preferred).
%           - amp_range : Amplitude value range in the form of [min, max].
%           - mode      : Color representation modes ('hsl' or 'hsv').
%           - reverse   : Whether reverse the color brightness.
% Output:   - img       : Color-coded representation.
%           - cbarimg   : Corresponding color bar.
% =========================================================================

amp = abs(cimg);        % amplitude of the input
pha = angle(cimg);      % phase of the input

amin = amp_range(1);    % minimum amplitude
amax = amp_range(2);    % maximum amplitude

% normalize amplitude values
amp = min(amp,amax);
amp = max(amp,amin);
amp_norm = (amp-amin)/(amax-amin);

ncmap = length(cmap);
img = zeros(size(cimg,1),size(cimg,2),3);

for i = 1:size(cimg,1)
    for j = 1:size(cimg,2)
        a = amp_norm(i,j);
        if reverse
            a = 1-a;
        end
        if strcmpi(mode,'hsv')
            img(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(pha(i,j)+pi)),:) .* a;
        elseif strcmpi(mode,'hsl')
            if a > 1/2
                w = 2*(a-1/2);
                img(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(pha(i,j)+pi)),:) .* (1-w) + [1,1,1] .* w;
            else
                w = 2*(1/2-a);
                img(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(pha(i,j)+pi)),:) .* (1-w) + [0,0,0] .* w;
            end
        end
        
    end
end

% draw the color bar
n = 256;
cbarimg = nan(n,n,3);
x = linspace(-1,1,n);
[X,Y] = meshgrid(x);
[theta,rho] = cart2pol(X,Y);
for i = 1:n
    for j = 1:n
        if rho(i,j) <= 1
            a = rho(i,j);
            if reverse
                a = 1-a;
            end
            if strcmpi(mode,'hsv')
                cbarimg(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(theta(i,j)+pi)),:) .* a;
            elseif strcmpi(mode,'hsl')
                if a > 1/2
                    w = 2*(a-1/2);
                    cbarimg(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(theta(i,j)+pi)),:) .* (1-w) + [1,1,1] .* w;
                else
                    w = 2*(1/2-a);
                    cbarimg(i,j,:) = cmap(1+round((ncmap-1)/2/pi*(theta(i,j)+pi)),:) .* (1-w) + [0,0,0] .* w;
                end
            end
        end
    end
end

end