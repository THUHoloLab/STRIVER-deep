% ========================================================================
% Post-processing code for holographic reconstruction:
%   - Merge the sliding window-based reconstruction into a sequence.
%   - Compensate for low-order aberrations via polynomial fitting.
%
% Author:   Yunhui Gao
% Date:     2025/06/10
% =========================================================================
%%
% =========================================================================
% load and merge sequences
% =========================================================================
clear;clc
close all

% select dataset
exp_num = 13;
grp_num = 97;

% select frames from the video
frame_start_vid = 70;
frame_end_vid = 129;

% define data batch size
K = 30;             % number of measurements within a batch
frame_step = 15;    % sliding window step size (typically K/2)

% select reconstruction algorithm
algorithm = 'tv';
w_probe   = true;

nullpixels_1 = 100;
nullpixels_2 = 100;

C1 = @(x) imgcrop(x,nullpixels_1);
C2 = @(x) imgcrop(x,nullpixels_2);

if w_probe
    str_prb = '_probe';
else
    str_prb = '';
end

pathname = ['data/exp/E',num2str(exp_num),'/G',num2str(grp_num),...
            '/results/results_',num2str(frame_start_vid),'_',...
            num2str(frame_end_vid),'/cache/'];
load([pathname,'results_',algorithm,str_prb,'_',num2str(frame_start_vid),'.mat']);

% crop the data to sensor FOV
x_crop = C2(C1(x_est(:,:,1)));
x = nan(size(x_crop,1),size(x_crop,2),frame_end_vid-frame_start_vid+1);

% loop over the sequences
for frame_start = frame_start_vid:frame_step:frame_end_vid+1-K
    fprintf('%4d -> %4d -> %4d \n',frame_start_vid, frame_start, frame_end_vid+1-K)
    load([pathname,'results_',algorithm,str_prb,'_',num2str(frame_start),'.mat'],'x_est');
    if frame_start == frame_start_vid
        for k = 1:K
            x(:,:,k+frame_start-frame_start_vid) = C2(C1(x_est(:,:,k)));
        end
    else
        % fade transition between neighboring windows
        for k = 1:floor(K/2)
            a = k/floor(K/2);
            x(:,:,k+frame_start-frame_start_vid) = (1-a) * x(:,:,k+frame_start-frame_start_vid) + a * C2(C1(x_est(:,:,k)));
        end
        for k = floor(K/2):K
            x(:,:,k+frame_start-frame_start_vid) = C2(C1(x_est(:,:,k)));
        end
    end
end

% =========================================================================
% display and save the merged sequence
% =========================================================================
figure,set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4],'color','w')
for k = 1:size(x,3)
    subplot(1,2,1),imshow(abs(x(:,:,k)),[0,1]);colorbar
    title('Retrieved sample amplitude','fontsize',10)
    ax = subplot(1,2,2);imshow(angle(x(:,:,k)),[-pi,pi]);colorbar
    colormap(ax,'inferno')
    title('Retrieved sample phase','fontsize',10)
    drawnow;
end

save([pathname,'results_',algorithm,str_prb,'_',num2str(frame_start_vid),'_',num2str(frame_end_vid),'.mat'],'x','rect_aoi_image','params');

%%
% =========================================================================
% compensate for aberration
% =========================================================================
clear;clc
close all

addpath(genpath('utils'))

% select dataset
exp_num = 13;
grp_num = 97;

% select frames from the video
frame_start_vid = 70;
frame_end_vid = 129;

% select reconstruction algorithm
algorithm = 'vidnet+tv';
w_probe   = true;

if w_probe
    str_prb = '_probe';
else
    str_prb = '';
end

% load data
pathname = ['data/exp/E',num2str(exp_num),'/G',num2str(grp_num),...
            '/results/results_',num2str(frame_start_vid),'_',...
            num2str(frame_end_vid),'/'];
filename = [pathname,'cache/','results_',algorithm,str_prb,'_',...
            num2str(frame_start_vid),'_',num2str(frame_end_vid),'.mat'];
load(filename,'x','rect_aoi_image','params')

% display data
figure,set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4],'color','w')
for k = 1:size(x,3)
    subplot(1,2,1),imshow(abs(x(:,:,k)),[0,1]);colorbar
    title('Retrieved sample amplitude','fontsize',10)
    ax = subplot(1,2,2);imshow(angle(x(:,:,k)),[-pi,pi]);colorbar
    colormap(ax,'inferno')
    title('Retrieved sample phase','fontsize',10)
    drawnow;
end

% =========================================================================
% phase compensation
% =========================================================================

max_ord = 4;    % maximum polynomial order
n_rect = 0;     % number of crop rectangles to select the background

% define or load background mask
mask_pha = [];
ref_filename = [pathname,'results_tv_',...
            num2str(frame_start_vid),'_',num2str(frame_end_vid),'.mat'];
load(ref_filename,'mask_pha');

% polynomial fitting
[A, xc, yc, mask_pha] = legendrebasis(max(angle(x),[],3), max_ord, n_rect, mask_pha);
amp = nan(size(x));
pha = nan(size(x));
for i = 1:size(x,3)
    amp(:,:,i) = abs(x(:,:,i));
    pha_temp = angle(x(:,:,i));
    v = mask_pha .* pha_temp;
    v = v(:);
    v(isnan(v)) = [];
    coefs = pinv(A)*v;
    bg = legendrepolyfit(xc, yc, max_ord, coefs);
    pha(:,:,i) = pha_temp - bg;
end

% display results
bias = 0.02;    figw = 0.90;    figh = 0.80;
figure,set(gcf,'unit','normalized','position',[(1-figw)/2,(1-figh)/2,figw,figh],'color','w')
[~, pos] = tight_subplot(1,2,[bias bias],[bias bias+0.04],[bias bias]);
for i = 1:size(x,3)
    ax = subplot(1,2,1);
    ax.Position = pos{1};
    imshow(amp(:,:,i),[0,1.2]);
    title('Retrieved amplitude','fontsize',10)
    ax = subplot(1,2,2);
    ax.Position = pos{2};
    imshow(angle(exp(1i*pha(:,:,i))),[]);
    title('Retrieved phase','fontsize',10)
    drawnow;
end

x = amp.*exp(1i*pha);

% =========================================================================
% amplitude compensation
% =========================================================================

max_ord = 1;
n_rect = 0;

% define or load background mask
mask_amp = [];
ref_filename = [pathname,'results_tv_',...
            num2str(frame_start_vid),'_',num2str(frame_end_vid),'.mat'];
load(ref_filename,'mask_amp');
[A, xc, yc, mask_amp] = legendrebasis(min(abs(x),[],3), max_ord, n_rect, mask_amp);

% polynomial fitting
amp = nan(size(x));
pha = nan(size(x));
for i = 1:size(x,3)
    pha(:,:,i) = angle(x(:,:,i));
    amp_temp = abs(x(:,:,i));
    v = mask_amp .* amp_temp;
    v = v(:);
    v(isnan(v)) = [];
    coefs = pinv(A)*v;
    bg = legendrepolyfit(xc, yc, max_ord, coefs);
    amp(:,:,i) = amp_temp./bg;
end

% display results
bias = 0.02;    figw = 0.90;    figh = 0.80;
figure,set(gcf,'unit','normalized','position',[(1-figw)/2,(1-figh)/2,figw,figh],'color','w')
[~, pos] = tight_subplot(1,2,[bias bias],[bias bias+0.04],[bias bias]);
for i = 1:size(x,3)
    ax = subplot(1,2,1);
    ax.Position = pos{1};
    imshow(amp(:,:,i),[0,1.2]);
    title('Retrieved amplitude','fontsize',10)
    ax = subplot(1,2,2);
    ax.Position = pos{2};
    imshow(angle(exp(1i*pha(:,:,i))),[]);
    title('Retrieved phase','fontsize',10)
    drawnow;
end

x = amp.*exp(1i*pha);
save(filename,'x','mask_amp','mask_pha','rect_aoi_image','params');

%%
% =========================================================================
% auxiliary functions
% =========================================================================

function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);
end
