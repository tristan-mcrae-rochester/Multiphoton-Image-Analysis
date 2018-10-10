clc; clear all; close all;


%{
load('ground.mat')
load('blurred_images.mat')
shape_images = size(blurred_images);
num_images = shape_images(1);
load in prediction images from python
deconvolved = zeros(shape_images);
my_scale = [5, 5];
%}
load('isonet.mat')
ground = double(isonet_ground);
input = isonet_input;
pred = double(isonet_preds);

psnr_results = zeros(44, 3);
val_indices = [0, 4, 30, 38, 23,  8, 34, 37, 15] +1;

for i = 1:44
    disp(i)
    %{
    input_lucy = squeeze(isonet_input(i, :, :));
    J_xy = deconvlucy(input_lucy, psf, 1000);
    blind_pred = deconvblind(input_lucy, ones(my_scale), 30);
    input_lucy = input_lucy(100:400, 100:400);
    ground_lucy = squeeze(ground_images(100:400, 100:400, i));
    pred_lucy = squeeze(J_xy(100:400, 100:400));
    input_isonet = squeeze(isonet_input(i, 100:400, 100:400));
    pred_isonet = squeeze(isonet_preds(i, 100:400, 100:400));
    ground_isonet = squeeze(ground_images(100:400, 100:400, i));
    blind_pred = squeeze(blind_pred(100:400, 100:400));
    %}
   
    input_im  = squeeze(input(i, :, :));
    pred_im   = squeeze(pred(i, :, :));
    ground_im = squeeze(ground(i, :, :));
    
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(input_im, [])
    psnr_im = psnr(input_im, ground_im);
    title(strcat("Input PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/8-31/input_slice_', int2str(i), '.png'))
    psnr_results(i, 1) = psnr_im;
    
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(pred_im, [])
    psnr_im = psnr(pred_im, ground_im);
    title(strcat("Pred PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/8-31/pred_slice_', int2str(i), '.png'))
    if ismember(i, val_indices)
        psnr_results(i, 3) = psnr_im;
    else   
        psnr_results(i, 2) = psnr_im;   
    end
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(ground_im, [])
    psnr_im = psnr(ground_im, ground_im);
    title(strcat("Ground PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/8-31/ground_slice_', int2str(i), '.png'))
    
    if ismember(i, val_indices)
        name = 'deconvolved/8-31/validation_comparison_slice_';
    else
        name = 'deconvolved/8-31/training_comparison_slice_';
    end
    
    fig = figure;set(gcf,'Visible', 'off');
    subplot(131)
    imshow(input_im, [])
    psnr_im = psnr(input_im, ground_im);
    title(strcat("Input PSNR = ", num2str(psnr_im)))
    subplot(132)
    imshow(pred_im, [])
    psnr_im = psnr(pred_im, ground_im);
    title(strcat("Pred PSNR = ", num2str(psnr_im)))
    subplot(133)
    imshow(ground_im, [])
    psnr_im = psnr(ground_im, ground_im);
    title(strcat("Ground PSNR = ", num2str(psnr_im)))

    saveas(fig, strcat(name, int2str(i), '.png'))
    
    
    %{
    
    fig = figure;
    imshow(pred_lucy)
    psnr_im = psnr(pred_lucy, ground_lucy);
    title(strcat("Lucy Pred PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/pred_lucy_slice_', int2str(i), '.png'))
    psnr_results(i, 1) = psnr_im;
    
    fig = figure;
    imshow(input_lucy)
    psnr_im = psnr(input_lucy, ground_lucy);
    title(strcat("Input PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/input_slice_', int2str(i), '.png'))
    
    fig = figure;
    imshow(pred_isonet)
    psnr_im = psnr(pred_isonet, ground_isonet);
    title(strcat("Isonet Pred PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/input_isonet_slice_', int2str(i), '.png'))
    psnr_results(i, 2) = psnr_im;
    
    fig = figure;
    imshow(ground_isonet)
    psnr_im = psnr(ground_isonet, ground_isonet);
    title(strcat("Ground PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/ground_slice_', int2str(i), '.png'))
    fig = figure;
    imshow(input_isonet)
    psnr_im = psnr(input_isonet, ground_isonet);
    title(strcat("Isonet Input PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/input_isonet_slice_', int2str(i), '.png'))
    fig = figure;
    imshow(ground_lucy)
    psnr_im = psnr(ground_lucy, ground_lucy);
    title(strcat("Lucy Ground PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/ground_lucy_slice_', int2str(i), '.png'))
    
    fig = figure;
    imshow(blind_pred)
    psnr_im = psnr(blind_pred, ground_lucy);
    title(strcat("Blind Pred PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/pred_blind_slice_', int2str(i), '.png'))
    %}
    
    %close all;
end



