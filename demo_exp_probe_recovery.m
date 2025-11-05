% ========================================================================
% This code provides a simple demonstration of the video plug-and-play 
% (PnP) algorithm for dynamic holographic reconstruction based on 
% experimental data.
%
% Author:   Yunhui Gao
% Date:     2025/06/10
% =========================================================================
%%
% =========================================================================
% data pre-processing
% =========================================================================
clear;clc;

save_results = true;

% load functions
addpath(genpath('utils'))

% select dataset
exp_num = 13;
grp_num = 97;

% select frames from the video
frame_start_vid = 70;
frame_end_vid = 129;

% define data batch size
K = 30;             % number of measurements within a batch
frame_step = 15;    % sliding window step size (typically K/2)

% loop over batches
for frame_start = frame_start_vid:frame_step:frame_end_vid-K+1

    close all
    
    % load diffuser and position calibration data
    foldername = ['data/exp/E',num2str(exp_num),'/G',num2str(grp_num)];
    load([foldername,'/calib/calib_diffuser.mat'],'diffuser','bias_1','bias_2','sizeout_1','sizeout_2','prefix','params');
    load([foldername,'/calib/calib_shift.mat'],   'shifts')
    
    % extract relative translation positions (total shift range)
    shifts_ref = shifts(:,frame_start_vid);
    shifts = shifts(:,frame_start:frame_start+K-1); % calibrated lateral displacement
    if sum(isnan(shifts(:)))
        error('Shifts include nan values.');
    end
    shifts = shifts - shifts_ref;
    
%     % select image area
%     fprintf('Please select the image area ... ')
%     img_obj = padimage(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp'])),...
%             [bias_1,bias_2],[sizeout_1,sizeout_2]);
%     figure
%     [temp,rect_aoi_image] = imcrop(img_obj);
%     if rem(size(temp,1),2) == 1
%         rect_aoi_image(4) = rect_aoi_image(4) - 1;
%     end
%     if rem(size(temp,2),2) == 1
%         rect_aoi_image(3) = rect_aoi_image(3) - 1;
%     end
%     close
%     fprintf('Selected. \n')

    % load pre-specified area of interest
    load([foldername,'/results/results_',num2str(frame_start_vid),'_', ...
          num2str(frame_end_vid),'/results_vidnet+tv_probe_',num2str(frame_start_vid),...
          '_',num2str(frame_end_vid),'.mat'],'rect_aoi_image')
    
    nullpixels_1 = 100;
    nullpixels_2 = 100;
    
    % spatial dimension of the raw image
    M1 = round(rect_aoi_image(4));
    M2 = round(rect_aoi_image(3));
    y = nan(M1,M2,K);
    
    % spatial dimension of the diffuser
    MM1 = M1 + 2*nullpixels_2;
    MM2 = M2 + 2*nullpixels_2;
    diffusers = nan(MM1,MM2,K);

    rect_aoi_diffuser = [rect_aoi_image(1)-nullpixels_2,   rect_aoi_image(2)-nullpixels_2,...
                         rect_aoi_image(3)+nullpixels_2*2, rect_aoi_image(4)+nullpixels_2*2];
    
    % calculate the model parameters for each measurement
    mask  = ones(size(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp']))));
    masks = zeros(M1,M2,K);
    for k = 1:K
        fprintf('Loading raw data: %02d / %02d\n', k, K)

        % pre-process the captured raw image
        img_obj = padimage(im2double(imread([foldername,'/',prefix, ...
                           num2str(frame_start+(k-1)),'.bmp'])),[bias_1,bias_2],[sizeout_1,sizeout_2]);
        img_obj  = abs(imshift(img_obj, shifts(1,k), shifts(2,k)));
        y(:,:,k) = imcrop(img_obj,rect_aoi_image);
        
        % define the binary mask (to avoid boundary artifact outside the sensor FOV)
        mask_tmp = padimage(mask,[bias_1,bias_2],[sizeout_1,sizeout_2]);
        mask_tmp = abs(imshift(mask_tmp, shifts(1,k), shifts(2,k)));
        masks(:,:,k) = imcrop(mask_tmp,rect_aoi_image);
        
        % calculate the translated diffuser profile
        diff = imshift(diffuser, shifts(1,k), shifts(2,k));
        diffusers(:,:,k) = imcrop(abs(diff),rect_aoi_diffuser) .* exp(1i*imcrop(angle(diff),rect_aoi_diffuser));
    end
    
    % display experimental data
    figure
    set(gcf,'unit','normalized','position',[0.15,0.25,0.7,0.5],'color','w')
    for k = 1:K
        subplot(1,3,1),imshow(y(:,:,k),[]);
        subplot(1,3,2),imshow(abs(diffusers(:,:,k)),[])
        subplot(1,3,3),imshow(masks(:,:,k),[0,1])
        drawnow;
    end
    
    % spatial dimension of the sample
    N1 = M1 + 2*nullpixels_2 + 2*nullpixels_1;  
    N2 = M2 + 2*nullpixels_2 + 2*nullpixels_1;
    
    % pre-calculate the transfer functions for diffraction modeling
    HQ1 = fftshift(transfunc_propagate(N1,N2, params.dist_1,params.pxsize,params.wavlen)); % forward propagation
    HQ2 = fftshift(transfunc_propagate(M1+2*nullpixels_2,M2+2*nullpixels_2, params.dist_2,params.pxsize,params.wavlen)); % forward propagation
    
    % forward model
    Q1  = @(x)   ifft2(fft2(x).*HQ1);                   % free-space propagation operator from sample to diffuser
    Q1H = @(x)   ifft2(fft2(x).*conj(HQ1));             % Hermitian operator of Q1
    C1  = @(x)   imgcrop(x,nullpixels_1);               % image cropping operator
    C1T = @(x)   zeropad(x,nullpixels_1);               % transpose operator of C1
    M   = @(x,k) x.*diffusers(:,:,k);                   % diffuser modulation operator
    MH  = @(x,k) x.*conj(diffusers(:,:,k));             % Hermitian operator of M
    Q2  = @(x)   ifft2(fft2(x).*HQ2);                   % free-space propagation operator from diffuser to sensor
    Q2H = @(x)   ifft2(fft2(x).*conj(HQ2));             % Hermitian operator of Q2
    C2  = @(x)   imgcrop(x,nullpixels_2);               % image cropping operator
    C2T = @(x)   zeropad(x,nullpixels_2);               % transpose operator of C2
    S   = @(x,k) x.*masks(:,:,k);                       % masking operator to avoid invalid pixels
    ST  = @(x,k) x.*conj(masks(:,:,k));                 % transpose operator of S
    A   = @(x,k) S(C2(Q2(M(C1(Q1(x)),k))),k);           % overall measurement operator
    AH  = @(x,k) Q1H(C1T(MH(Q2H(C2T(ST(x,k))),k)));     % Hermitian operator of A
    
    % =========================================================================
    % reconstruction algorithm
    % =========================================================================
    
    gpu = 1;        % whether using GPU or not
    
    K_recon = K;    % number of reconstructed frames
    
    x_est = ones(M1+2*nullpixels_1+2*nullpixels_2,M2+2*nullpixels_1+2*nullpixels_2,K_recon);     % initial guess
    
    % regularization parameters
    lams_s = [1e-1, 2e-3];      % regularization parameter (spatial)
    lams_t = [1e-1, 3e-3];      % regularization parameter (temporal)
    
    % algorithm parameters
    alpha = 10;                 % hyperparameter for tuning regularization weights
    gam = 2;                    % step size (see the paper for details)
    n_iters = 200;              % number of iterations (main loop)
    n_subiters = 10;            % number of subiterations (proximal update)
    
    % auxilary variables
    z_est = x_est;
    g_est = zeros(size(x_est));
    v_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);
    w_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);
    
    % auxilary functions
    
    lams = @(iter) reg_param(iter, n_iters, alpha, ...
                             [lams_s(1),lams_s(1),lams_t(1)], ...
                             [lams_s(2),lams_s(2),lams_t(2)]);
                                % define the regularization parameters at each iteration
    KK   = @(k) ceil(K_recon*k/K);   % round to integer indices
    
    % initialize GPU
    if gpu
        device    = gpuDevice(gpu);
        reset(device)
        x_est     = gpuArray(x_est);
        y         = gpuArray(y);
        HQ1       = gpuArray(HQ1);
        HQ2       = gpuArray(HQ2);
        diffusers = gpuArray(diffusers);
        masks     = gpuArray(masks);
        g_est     = gpuArray(g_est);
        z_est     = gpuArray(z_est);
        v_est     = gpuArray(v_est);
        w_est     = gpuArray(w_est);
    end
    
    % main loop
    timer = tic;
    for iter = 1:n_iters
    
        % gradient update
        g_est(:) = 0;
        for k = 1:K
            u = A(z_est(:,:,KK(k)),k);
            u = (abs(u) - sqrt(y(:,:,k))) .* exp(1i*angle(u));
            g_est(:,:,KK(k)) = g_est(:,:,KK(k)) + 1/2/(K/K_recon) * AH(u,k);
        end
        u = z_est - gam * g_est;
    
        % proximal update
        v_est(:) = 0; w_est(:) = 0;
        [lam_1,lam_2,lam_3] = lams(iter);
        for subiter = 1:n_subiters
            w_next = v_est + 1/12/gam*Df(u-gam*DTf(v_est));
            w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_1).*exp(1i*angle(w_next(:,:,:,1)));
            w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_2).*exp(1i*angle(w_next(:,:,:,2)));
            w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_3).*exp(1i*angle(w_next(:,:,:,3)));
            v_est = w_next + subiter/(subiter+3)*(w_next-w_est);
            w_est = w_next;
        end
        x_next = u - gam*DTf(w_est);
        
        % Nesterov extrapolation
        z_est = x_next + (iter/(iter+3))*(x_next - x_est);
        
        x_est = x_next;

        % print status
        runtime = toc(timer);
        fprintf('iter: %4d / %4d | runtime: %7.2f s\n', iter, n_iters, runtime);
    end
    
    % wait for GPU
    if gpu; wait(device); end
    
    % gather data from GPU
    if gpu
        x_est     = gather(x_est);
        y         = gather(y);
        HQ1       = gather(HQ1);
        HQ2       = gather(HQ2);
        diffusers = gather(diffusers);
        masks     = gather(masks);
    end
    
    % =====================================================================
    % display results
    % =====================================================================
    figure
    set(gcf,'unit','normalized','position',[0.2,0.25,0.6,0.5],'color','w')
    for k = 1:K
        ax = subplot(1,2,1);imshow(abs(C2(C1(x_est(:,:,KK(k))))),[0,1]);colorbar
        colormap(ax,'gray')
        title('Retrieved sample amplitude','fontsize',10)
        ax = subplot(1,2,2);imshow(angle(C2(C1(exp(-1i*2)*x_est(:,:,KK(k))))),[-pi,pi]);colorbar
        colormap(ax,'inferno')
        title('Retrieved sample phase','fontsize',10)
        drawnow;
    end
    
    % save results as a .mat file
    if save_results
        pathname = [foldername,'/results/results_',num2str(frame_start_vid),'_',num2str(frame_end_vid),'/cache/']; 
        if ~isfolder(pathname); mkdir(pathname); end
        save([pathname,'results_tv_',num2str(frame_start),'.mat'],'x_est','rect_aoi_image','params')
    end
end

%%
% =========================================================================
% initialize probe
% =========================================================================
close all
pathname = [foldername,'/results/results_',num2str(frame_start_vid),'_',num2str(frame_end_vid),'/cache/']; 
x_est_all = zeros(size(x_est,1),size(x_est,2),frame_end_vid-frame_start_vid+1);

% load the entire reconstructed video sequence
for frame_start = frame_start_vid:frame_step:frame_end_vid+1-K
    fprintf('%4d -> %4d -> %4d \n',frame_start_vid, frame_start, frame_end_vid+1-K)
    load([pathname,'results_tv_',num2str(frame_start),'.mat'],'x_est');
    if frame_start == frame_start_vid
        for k = 1:K
            x_est_all(:,:,k+frame_start-frame_start_vid) = x_est(:,:,k);
        end
    else
        for k = 1:floor(K/2)
            a = k/floor(K/2);
            x_est_all(:,:,k+frame_start-frame_start_vid) = ...
                (1-a) * x_est_all(:,:,k+frame_start-frame_start_vid) + a * x_est(:,:,k);
        end
        for k = floor(K/2):K
            x_est_all(:,:,k+frame_start-frame_start_vid) = x_est(:,:,k);
        end
    end
end

% load diffuser and position calibration data
load([foldername,'/calib/calib_diffuser.mat'],'diffuser','bias_1','bias_2','sizeout_1','sizeout_2','prefix','params');
load([foldername,'/calib/calib_shift.mat'],   'shifts')

% calibrated lateral shifts
shifts_ref = shifts(:,frame_start_vid);
shifts = shifts(:,frame_start_vid:frame_end_vid);
shifts = shifts - shifts_ref;

th = max(abs(x_est_all(:)))+1;
x_est_tmp = x_est_all;

shift_range_1 = shifts(1,end);
shift_range_2 = shifts(2,end);
padsize_1 = ceil(abs(shift_range_1));
padsize_2 = ceil(abs(shift_range_2));
if rem(padsize_1,2) == 1
    padsize_1 = padsize_1+1;
end
if rem(padsize_2,2) == 1
    padsize_2 = padsize_2+1;
end

if shift_range_1 > 0
    P1  = @(x) padarray(x,[0,padsize_1],th,'pre');
    C3  = @(x) x(:,padsize_1+1:end);
    C3T = @(x) padarray(x,[0,padsize_1],0,'pre');
else
    P1  = @(x) padarray(x,[0,padsize_1],th,'post');
    C3 = @(x) x(:,1:end-padsize_1);
    C3T = @(x) padarray(x,[0,padsize_1],0,'post');
end
if shift_range_2 > 0
    P2  = @(x) padarray(x,[padsize_2,0],th,'pre');
    C4  = @(x) x(padsize_2+1:end,:);
    C4T = @(x) padarray(x,[padsize_2,0],0,'pre');
else
    P2  = @(x) padarray(x,[padsize_2,0],th,'post');
    C4  = @(x) x(1:end-padsize_2,:);
    C4T = @(x) padarray(x,[padsize_2,0],0,'post');
end

x_est_tmp = P2(P1(x_est_tmp));
x_est_shift = zeros(size(x_est_tmp));
for k = 1:size(shifts,2)
    x_est_shift(:,:,k) = imshift(x_est_tmp(:,:,k),-shifts(1,k),-shifts(2,k));
end

x_est_shift(abs(x_est_shift)>th-0.5) = nan;

figure
for k = 1:frame_end_vid-frame_start_vid+1
    imshow(abs(x_est_shift(:,:,k)),[0,1]);
    drawnow
end

p_est = median(abs(x_est_shift),3,'omitnan');
p_est(isnan(p_est)) = 0;
p_est = imfilter(p_est,fspecial('gaussian',[10,10],5));

mirror_padsize_1 = 200;
mirror_padsize_2 = 270;
p_est_center = p_est(mirror_padsize_1+1:end-mirror_padsize_1,mirror_padsize_2+1:end-mirror_padsize_2);
p_est = padarray(p_est_center,[mirror_padsize_1,mirror_padsize_2],'symmetric');

figure,imshow(p_est,[0,1])

save([pathname,'probe_cache.mat'],'p_est')

%%
% =========================================================================
% blind ptychographic recovery
% =========================================================================
gpu = 1;

% rPIE parameters
alph_x = 1;
alph_p = 1;
step_x = 1;
step_p = 1;

a = 0.0;   % smoothness parameter
gam = 2;   % step size

figure
n_iters_refine = 100; % iteration number    

for frame_start = frame_start_vid:frame_step:frame_end_vid+1-K
    
    close all

    % =========================================================================
    % data pre-processing
    % =========================================================================
    
    % load diffuser and position calibration data
    load([foldername,'/calib/calib_diffuser.mat'],'diffuser','bias_1','bias_2','sizeout_1','sizeout_2','prefix','params');
    load([foldername,'/calib/calib_shift.mat'],'shifts')
    
    % calibrated lateral displacement
    shifts_ref = shifts(:,frame_start_vid);
    shifts = shifts(:,frame_start:frame_start+K-1);
    shifts = shifts - shifts_ref;
    img_obj = padimage(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp'])),...
            [bias_1,bias_2],[sizeout_1,sizeout_2]);
    
    % load pre-specified area of interest
    load([foldername,'/results/results_',num2str(frame_start_vid),'_',num2str(frame_end_vid), ...
          '/results_vidnet+tv_probe_',num2str(frame_start_vid),'_',num2str(frame_end_vid),'.mat'], ...
          'rect_aoi_image')

    nullpixels_1 = 100;
    nullpixels_2 = 100;
    
    % spatial dimension of the image
    M1 = round(rect_aoi_image(4));
    M2 = round(rect_aoi_image(3));
    y = nan(M1,M2,K);
    
    % spatial dimension of the diffuser
    MM1 = M1 + 2*nullpixels_2;
    MM2 = M2 + 2*nullpixels_2;
    diffusers = nan(MM1,MM2,K);
    rect_aoi_diffuser = [rect_aoi_image(1)-nullpixels_2, rect_aoi_image(2)-nullpixels_2,...
        rect_aoi_image(3)+2*nullpixels_2, rect_aoi_image(4)+2*nullpixels_2];
    
    % calculate the model parameters for each measurement
    mask  = ones(size(im2double(imread([foldername,'/',prefix,num2str(frame_start),'.bmp']))));
    masks = zeros(M1,M2,K);
    for k = 1:K
        fprintf('Loading raw data: %02d / %02d\n', k, K)
        
        % pre-process the captured raw image
        img_obj = padimage(im2double(imread([foldername,'/',prefix,num2str(frame_start+(k-1)),'.bmp'])),...
            [bias_1,bias_2],[sizeout_1,sizeout_2]);
        img_obj  = abs(imshift(img_obj, shifts(1,k), shifts(2,k)));
        y(:,:,k) = imcrop(img_obj,rect_aoi_image);
        
        % define the binary mask (to avoid boundary artifact outside the sensor FOV)
        mask_tmp = padimage(mask,[bias_1,bias_2],[sizeout_1,sizeout_2]);
        mask_tmp = abs(imshift(mask_tmp, shifts(1,k), shifts(2,k)));
        masks(:,:,k) = imcrop(mask_tmp,rect_aoi_image);
        
        % calculate the translated diffuser profile
        diff = imshift(diffuser, shifts(1,k), shifts(2,k));
        diffusers(:,:,k) = imcrop(abs(diff),rect_aoi_diffuser) .* exp(1i*imcrop(angle(diff),rect_aoi_diffuser));
    end
    
    % spatial dimension of the sample
    N1 = M1 + 2*nullpixels_2 + 2*nullpixels_1;  
    N2 = M2 + 2*nullpixels_2 + 2*nullpixels_1;
    
    % pre-calculate the transfer functions for diffraction modeling
    HQ1 = fftshift(transfunc_propagate(N1,N2, params.dist_1,params.pxsize,params.wavlen)); % forward propagation
    HQ2 = fftshift(transfunc_propagate(M1+2*nullpixels_2,M2+2*nullpixels_2, params.dist_2,params.pxsize,params.wavlen)); % forward propagation
    
    % forward model
    Q1  = @(x)   ifft2(fft2(x).*HQ1);                   % free-space propagation operator from sample to diffuser
    Q1H = @(x)   ifft2(fft2(x).*conj(HQ1));             % Hermitian operator of Q1
    C1  = @(x)   imgcrop(x,nullpixels_1);               % image cropping operator
    C1T = @(x)   zeropad(x,nullpixels_1);               % transpose operator of C1
    M   = @(x,k) x.*diffusers(:,:,k);                   % diffuser modulation operator
    MH  = @(x,k) x.*conj(diffusers(:,:,k));             % Hermitian operator of M
    Q2  = @(x)   ifft2(fft2(x).*HQ2);                   % free-space propagation operator from diffuser to sensor
    Q2H = @(x)   ifft2(fft2(x).*conj(HQ2));             % Hermitian operator of Q2
    C2  = @(x)   imgcrop(x,nullpixels_2);               % image cropping operator
    C2T = @(x)   zeropad(x,nullpixels_2);               % transpose operator of C2
    S   = @(x,k) x.*masks(:,:,k);                       % masking operator to avoid invalid pixels
    ST  = @(x,k) x.*conj(masks(:,:,k));                 % transpose operator of S
    A   = @(x,k) S(C2(Q2(M(C1(Q1(x)),k))),k);           % overall measurement operator
    AH  = @(x,k) Q1H(C1T(MH(Q2H(C2T(ST(x,k))),k)));     % Hermitian operator of A
    
    % load probe estimate
    load([pathname,'probe_cache.mat'],'p_est')
    p_est = p_est / median(abs(p_est(:)));
    
    % load TV-based reconstruction
    load([pathname,'results_tv_',num2str(frame_start),'.mat'],'x_est')

    shifts_ref = shifts;

    % auxiliary variables
    q_est = zeros(size(p_est));
    g_est = zeros(size(x_est));
    v_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);
    w_est = zeros(size(x_est,1),size(x_est,2),size(x_est,3),3);
    
    % initialize GPU
    if gpu
        device    = gpuDevice(gpu);
        reset(device)
        x_est     = gpuArray(x_est);
        p_est     = gpuArray(p_est);
        q_est     = gpuArray(q_est);
        y         = gpuArray(y);
        HQ1       = gpuArray(HQ1);
        HQ2       = gpuArray(HQ2);
        masks     = gpuArray(masks);
        diffusers = gpuArray(diffusers);
        g_est     = gpuArray(g_est);
        v_est     = gpuArray(v_est);
        w_est     = gpuArray(w_est);
    end
    
    T   = @(x,posi) imshift(x, posi(1), posi(2));       % lateral translation operator
    TH  = @(x,posi) imshift(x,-posi(1),-posi(2));       % Hermitian of T
    C0  = @(x) C4(C3(x));
    C0T = @(x) C3T(C4T(x));
    
    % =========================================================================
    % main loop
    % =========================================================================
    timer = tic;
    for iter = 1:n_iters_refine
        
        g_est(:) = 0; q_est(:) = 0;

        for k = 1:K
            
            % translation position
            pos = shifts_ref(:,k);
            
            % translation operator
            G  = @(x) C0(T(x,pos));     
            GH = @(x) TH(C0T(x),pos);
            
            % probe and object function
            p_crp = G(p_est);
            x_est_k = x_est(:,:,k);
            
            % exit wave
            u = x_est_k.*p_crp;
            
            % gradient calculation
            u = A(u,k);
            u = (abs(u) - sqrt(y(:,:,k))) .* exp(1i*angle(u));
            u = AH(u,k);
            d_x = conj(p_crp) .* u;
            d_p = conj(x_est_k) .* u;
            
            % rPIE
            weight_x = 1 ./ ( (1-alph_x) .* abs(p_crp).^2 + alph_x .* max(abs(p_crp(:)).^2)  );
            weight_p = 1 ./ ( (1-alph_p) .* abs(x_est_k).^2  + alph_p .* max(abs(x_est_k(:))).^2   );
            g_est(:,:,k) = g_est(:,:,k) + d_x .* weight_x;
            q_est = q_est + GH(d_p .* weight_p);
            
        end
        
        % gradient update
        q_est = q_est / K + a * DTf(Df(p_est));
        p_est = p_est - step_p * q_est;
        x_est = x_est - step_x * g_est;
        
        % proximal update
        v_est(:) = 0; w_est(:) = 0;
        [lam_1,lam_2,lam_3] = lams(n_iters_refine);
        for subiter = 1:n_subiters
            w_next = v_est + 1/12/gam*Df(x_est-gam*DTf(v_est));
            w_next(:,:,:,1) = min(abs(w_next(:,:,:,1)),lam_1).*exp(1i*angle(w_next(:,:,:,1)));
            w_next(:,:,:,2) = min(abs(w_next(:,:,:,2)),lam_2).*exp(1i*angle(w_next(:,:,:,2)));
            w_next(:,:,:,3) = min(abs(w_next(:,:,:,3)),lam_3).*exp(1i*angle(w_next(:,:,:,3)));
            v_est = w_next + subiter/(subiter+3)*(w_next-w_est);
            w_est = w_next;
        end
        x_est = x_est - gam*DTf(w_est);
        
        % print status
        runtime = toc(timer);
        fprintf('iter: %4d / %4d | runtime: %7.2f s\n', iter, n_iters_refine, runtime);
    end
    
    % wait for GPU
    if gpu; wait(device); end
    
    % gather data from GPU
    if gpu
        x_est  = gather(x_est);
        p_est  = gather(p_est);
        y      = gather(y);
        HQ1    = gather(HQ1);
        HQ2    = gather(HQ2);
        masks  = gather(masks);
        diffusers = gather(diffusers);
    end
    

    % =====================================================================
    % Display results
    % =====================================================================
    figure
    set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4],'color','w')
    for k = 1:K
        subplot(1,2,1),imshow(abs(C2(C1(x_est(:,:,KK(k))))),[0,1]);colorbar
        title('Retrieved sample amplitude','fontsize',10)
        ax = subplot(1,2,2);imshow(angle(C2(C1(exp(-1i*2)*x_est(:,:,KK(k))))),[-pi,pi]);colorbar
        colormap(ax,'inferno')
        title('Retrieved sample phase','fontsize',10)
        drawnow;
    end

    if save_results
        pathname = [foldername,'/results/results_',num2str(frame_start_vid),'_',num2str(frame_end_vid),'/cache/']; 
        if ~isfolder(pathname); mkdir(pathname); end
        save([pathname,'results_tv_probe_',num2str(frame_start),'.mat'],'x_est','rect_aoi_image','params','p_est')
    end
end

%%
% =========================================================================
% Auxiliary functions
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


function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);
end


function [lam_1, lam_2, lam_3] = reg_param(iter,n_iters,alpha,lams_start,lams_end)
% =========================================================================
% Calculate the regularization parameters at each iteration, with 
% exponentially decaying values.
% -------------------------------------------------------------------------
% Input:    - iter              : The current iteration number.
%           - n_iters           : Total iteration numbers.
%           - alpha             : Decaying parameter.
%           - lams_start        : Initial regularization parameter.
%           - lams_end          : Final regularization parameter.
% Output:   - lam_1,lam_2,lam_3 : The current regularization parameter for the three dimensions.
% =========================================================================

lam_1 = lams_end(1) + (lams_start(1) - lams_end(1)) * exp(-alpha * (iter-1)/n_iters);
lam_2 = lams_end(2) + (lams_start(2) - lams_end(2)) * exp(-alpha * (iter-1)/n_iters);
lam_3 = lams_end(3) + (lams_start(3) - lams_end(3)) * exp(-alpha * (iter-1)/n_iters);

end


function H = transfunc_propagate(n1, n2, dist, pxsize, wavlen)
% =========================================================================
% Calculate the transfer function of the free-space diffraction.
% -------------------------------------------------------------------------
% Input:    - n1, n2   : The image dimensions (pixel).
%           - dist     : Propagation distance.
%           - pxsize   : Pixel (sampling) size.
%           - wavlen   : Wavelength of the light.
% Output:   - H        : Transfer function.
% =========================================================================

% sampling in the spatial frequency domain
k1 = pi/pxsize*(-1:2/n1:1-2/n1);
k2 = pi/pxsize*(-1:2/n2:1-2/n2);
[K2,K1] = meshgrid(k2,k1);

k = 2*pi/wavlen;    % wave number

ind = (K1.^2 + K2.^2 >= k^2);  % remove evanescent orders
K1(ind) = 0; K2(ind) = 0;

H = exp(1i*dist*sqrt(k^2-K1.^2-K2.^2));

end


function H = transfunc_imshift(n1,n2,s1,s2)
% =========================================================================
% Calculate the transfer function of the image shifting operation.
% -------------------------------------------------------------------------
% Input:    - n1, n2   : Image dimension (pixel).
%           - s1, s2   : Shifts along the two dimensions (pixel).
% Output:   - H        : Transfer function.
% =========================================================================
f1 = -n1/2:1:n1/2-1;
f2 = -n2/2:1:n2/2-1;
[u2,u1] = meshgrid(f2,f1);

H = exp(-1i*2*pi*(s1*u1/n1 + s2*u2/n2));

end


function w = Df(x)
% =========================================================================
% Calculate the 3D gradient (finite difference) of an input 3D datacube.
% -------------------------------------------------------------------------
% Input:    - x  : The input 3D datacube.
% Output:   - w  : The gradient (4D array).
% =========================================================================
if size(x,3) > 1
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               x(:,:,1:end) - x(:,:,[2:end,end]));
else
    w = cat(4, x(1:end,:,:) - x([2:end,end],:,:), ...
               x(:,1:end,:) - x(:,[2:end,end],:), ...
               zeros(size(x(:,:,1))));
end
end


function u = DTf(w)
% =========================================================================
% Calculate the transpose of the gradient operator.
% -------------------------------------------------------------------------
% Input:    - w  : 4D array.
% Output:   - x  : 3D array.
% =========================================================================
u1 = w(:,:,:,1) - w([end,1:end-1],:,:,1);
u1(1,:,:) = w(1,:,:,1);
u1(end,:,:) = -w(end-1,:,:,1);

u2 = w(:,:,:,2) - w(:,[end,1:end-1],:,2);
u2(:,1,:) = w(:,1,:,2);
u2(:,end,:) = -w(:,end-1,:,2);

if size(w,3) > 1
    u3 = w(:,:,:,3) - w(:,:,[end,1:end-1],3);
    u3(:,:,1) = w(:,:,1,3);
    u3(:,:,end) = -w(:,:,end-1,3);
else
    u3 = 0;
end

u = u1 + u2 + u3;

end
