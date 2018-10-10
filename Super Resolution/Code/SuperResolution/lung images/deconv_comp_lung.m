clc; clear all; close all;


load('isonet.mat')
ground = double(isonet_ground);
input = isonet_input;
pred = double(isonet_preds);

psnr_results = zeros(103, 2);
%val_indices = [0, 4, 30, 38, 23,  8, 34, 37, 15] +1;

for i = 1:103
    disp(i)

    input_im  = squeeze(input(i, :, :));
    pred_im   = squeeze(pred(i, :, :));
    ground_im = squeeze(ground(i, :, :));
    
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(input_im, [0 0.0625])
    psnr_im = psnr(input_im, ground_im, max(ground_im(:)));
    title(strcat("Input PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/9-5/input_slice_', int2str(i), '.png'))
    psnr_results(i, 1) = psnr_im;
    
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(pred_im, [0 0.0625])
    psnr_im = psnr(pred_im, ground_im, max(ground_im(:)));
    title(strcat("Pred PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/9-5/pred_slice_', int2str(i), '.png'))
    %if ismember(i, val_indices)
    %    psnr_results(i, 3) = psnr_im;
    %else   
    psnr_results(i, 2) = psnr_im;   
    %end
    fig = figure;set(gcf,'Visible', 'off'); 
    imshow(ground_im, [0 0.0625])
    psnr_im = psnr(ground_im, ground_im, max(ground_im(:)));
    title(strcat("Ground PSNR = ", num2str(psnr_im)))
    saveas(fig, strcat('deconvolved/9-5/ground_slice_', int2str(i), '.png'))
    
    %if ismember(i, val_indices)
    %    name = 'deconvolved/9-5/validation_comparison_slice_';
    %else
    name = 'deconvolved/9-5/training_comparison_slice_';
    %end
    
    fig = figure;set(gcf,'Visible', 'off');
    subplot(131)
    imshow(input_im, [0 0.0625])
    psnr_im = psnr(input_im, ground_im, max(ground_im(:)));
    title(strcat("Input PSNR = ", num2str(psnr_im)))
    subplot(132)
    imshow(pred_im, [0 0.0625])
    psnr_im = psnr(pred_im, ground_im, max(ground_im(:)));
    title(strcat("Pred PSNR = ", num2str(psnr_im)))
    subplot(133)
    imshow(ground_im, [0 0.0625])
    psnr_im = psnr(ground_im, ground_im, max(ground_im(:)));
    title(strcat("Ground PSNR = ", num2str(psnr_im)))

    saveas(fig, strcat(name, int2str(i), '.png'))
    
end
