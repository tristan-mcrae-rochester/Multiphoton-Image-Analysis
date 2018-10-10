clc; clear all; close all;

load('psf_bead.mat'); %8.5 micrometers in 202 pixels 0.0421 microns per pixel
%the axial data here is 2 microns per pixel
%Therefore this PSF should be reduced to 4.25 pixels per 202 pixels here
%one pixel for every 47.5059 pixels
%three pixels total covering the central 96 pixels or 144?


fname =   'LungMAP_200um_1_zoom-in.tif' ;
info = imfinfo(fname);
num_images = numel(info);
num_channels = 3;
im_per_ch = num_images/num_channels;
my_scale = [512, 512];
microns_per_pixel = 0.5;
microns_per_voxel = 2;

fullstack = zeros([my_scale,num_images]);
ground = zeros([my_scale,im_per_ch]);
axial_scale = 4;
%axial_slice = 63
lateral_slice = 250
iter = 20;

disp('Reading')
for k = 1:num_images
    A = im2double(imread(fname, k));
    fullstack(:,:,k) = A;
end

ch3_indices = linspace(3, 309, im_per_ch);
ch2_indices = linspace(2, 308, im_per_ch);
ch1_indices = linspace(1, 307, im_per_ch);

ground = fullstack(:, :, ch3_indices);

%save('ground_lung.mat','ground','-v6');

psf_zy = transpose(psf_zy);
psf_exaggerated = psf_zy;
psf_zy = psf_zy(:, 10:153);
psf_zy_scaled = downsample(psf_zy, 48);
psf_zy_scaled = psf_zy_scaled;
psf = [sum(psf_zy(1:48)) sum(psf_zy(49:97)) sum(psf_zy(98:144))];
psf = [0.1 0.8 0.1]
%psf = psf_exaggerated

%{
isonet_axial = zeros([512, 512, 412]);%ground;
for x = 1:512
    disp(x)
    for y = 1:512
        isonet_axial(y, x, :) = interp(ground(y, x, :), axial_scale);
    end
end
isonet_axial = permute(isonet_axial,[1 , 3, 2]);
%}
load('lung.mat')

isonet_inputs = zeros(512, 412, im_per_ch);
isonet_labels = ground(:, 50:461, :);

figure
for z = 1:im_per_ch
    disp(z)
    subplot(221)
    orig_img = isonet_labels(:, :, z);
    imshow(orig_img, [0 0.0625])
    title('Original')
    
    subplot(222)
    blurred_image = conv2(orig_img, psf, 'same');
    imshow(blurred_image, [0 0.0625])
    title('Convolved')
    
    subplot(223)
    downsampled = zeros(512, im_per_ch);
    for x = 1:im_per_ch
        downsampled(:, x) = blurred_image(:, x*axial_scale);
    end
    lateral_downsampled = downsampled;
    
    upsampled = zeros(512, 412);
    %upsampled_old = zeros(512, 412);
    for x = 1:512
        %upsampled_old(x, :) = interp(downsampled(x, :), axial_scale);
        upsampled(x, :) = interp1(1:4:412, downsampled(x, :), 1:1:412, 'pchip');
    end
    imshow(upsampled, [0 0.0625])
    title('Downsampled')
    isonet_inputs(:, :, z) = upsampled;
    
    %subplot(224)
    %imshow(upsampled_old,  [0 0.0625])
    %title('Downsampled Old')
    %disp('New PSNR')
    %disp(psnr(upsampled, isonet_labels(:, :, z), 0.0625))
    %disp('Old PSNR')
    %disp(psnr(upsampled_old, isonet_labels(:, :, z), 0.0625))
    subplot(224)
    axial_im = squeeze(ground(:, lateral_slice, :));
    
    upsampled = zeros(512, 412);
    for y = 1:512
        upsampled(y, :) = interp1(1:4:412, axial_im(y, :), 1:1:412, 'pchip');
        %upsampled(y, :) = interp(axial_im(y, :), axial_scale);
    end
    imshow(upsampled, [0 0.0625])
    title('Axial Comp')
    
    ground_slice =  isonet_labels(:, :, z);
    max_val = max(ground_slice(:));
    disp(psnr(isonet_inputs(:, :, z), isonet_labels(:, :, z), max_val))
end

save('lung_cubic.mat','isonet_inputs', 'isonet_labels', 'isonet_axial', '-v6');


%{
figure
axial_deconv = deconvlucy(upsampled, psf, iter);
imshow(axial_deconv, [0 0.0625])
title('Lucy Deconv on Axial Image')
figure
imshow(orig_img, [0 0.0625])
title('Lateral Image')
figure
subplot(121)
imshow(lateral_downsampled, [0 0.0624])
title('Lateral Downsampled')
subplot(122)
imshow(axial_im, [0 0.0624])
title('Axial Downsampled')

figure

[blind psf] = deconvblind(upsampled, ones([10, 20]), iter);
subplot(121)
imshow(upsampled, [0 0.0625])
subplot(122)
imshow(blind, [0 0.0625])

figure
subplot(221)
imshow(psf, [])
title('PSF Upsampled Axial')

subplot(222)
imshow(psf_ground, [])
title('PSF Upsampled Lateral')

subplot(223)
imshow(psf_downsampled_axial, [])
title('PSF Downsampled Axial')

subplot(224)
imshow(psf_downsampled_ground, [])
title('PSF Downsampled Lateral')

%}



