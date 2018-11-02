psf_x = feval(PSFstats.fitStats.X.fit, PSFstats.fitStats.X.data.xVals);
psf_y = feval(PSFstats.fitStats.Y.fit, PSFstats.fitStats.Y.data.yVals);
psf_xy = psf_x*transpose(psf_y);

figure
subplot(121)
imshow(PSFstack(:, :, 36), [])
subplot(122)
imshow(psf_xy, [])
test_deconv = deconvlucy(PSFstack(:, :, 36), psf_xy);
imshow(test_deconv, [])

psf_zx = feval(PSFstats.fitStats.ZX.fit, PSFstats.fitStats.Y.data.yVals);

%clc; clear all; close all;

fname =   '36797.tif' 
info = imfinfo(fname);
num_images = numel(info);
my_scale = [475,476]
fullstack = zeros([my_scale,num_images]);
ground = zeros([my_scale,num_images/2]);

disp('Reading')
for k = 1:num_images
    A = im2double(imread(fname, k));
    A = imresize(A , my_scale);
    fullstack(:,:,k) = A;
end

stack_indices = linspace(2, 88, 44);
ground_indices = linspace(1, 87, 44);
stack = fullstack(:, :, stack_indices);
ground = fullstack(:, :, ground_indices);

save('ground.mat','ground','-v6');

blurred = zeros([my_scale,num_images/2]);

psf = psf_zx;

disp('Blurring')
for i = 1:num_images/2
    ground_image = ground(:, :, i);
    blurred_image = conv2(ground_image, psf, 'same');
    blurred(:, :, i) = blurred_image;
end

disp('Downsampling')
downsampled = zeros(237, 476, num_images/2);
for z = 1:num_images/2
    for x = 1:237
        downsampled(x, :, z) = blurred(x*2, :, z);
    end
end

save('pseudo_observed.mat', 'downsampled', '-v6')

downsampled_psf = psf(1:2:201);
deconvolved = zeros(237, 476, num_images/2);
disp('Deconvolving')
for z = 1:num_images/2
    disp(z)
    downsampled_image = downsampled(:, :, z);
    deconvolved(:, :, z) = squeeze(deconvlucy(downsampled_image, downsampled_psf, 1000));    
    %figure
    %imshow(deconvolved(:, :, z), [])
end

save('deconvolved.mat', 'deconvolved', '-v6')
upsampled = zeros([my_scale,num_images/2]);
disp('Upsampling')
for z = 1:num_images/2
    for y = 1:476
        deconvolved_array = deconvolved(:, y, z);
        upsampled(1:474, y, z) = interp(deconvolved_array, 2);
    end
end


save('interpolation_upsampled.mat', 'upsampled', '-v6')



figure
subplot(221)
imshow(blurred(:, :, 16), [])
subplot(222)
imshow(downsampled(:, :, 16), [])
subplot(223)
imshow(deconvolved(:, :, 16), [])
subplot(224)
imshow(upsampled(:, :, 16), [])



