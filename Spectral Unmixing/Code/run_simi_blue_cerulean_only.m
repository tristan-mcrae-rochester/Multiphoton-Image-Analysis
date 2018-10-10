clc; clear all; close all;

addpath('D:\Projects\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif' %
num_fluorophores = 2;
num_channels = 5;
multi_fluorophore_pixels = false;
mixed_image = read_tiff(num_channels, load_loc, 10, 11)*16;
%Available fluorophores
%"RFP (TagRFP)"
%"Fluorescein (FITC)"
%"DAPI"
%"SHG"
%"eGFP"
%"DsRed"
%"CFP"
%"Cerulean"
%"eBFP"
%"iRFP670"
%"mAzamiGreen"
%"mCherry"
%"Citrine"

%fluorophores = ["RFP (TagRFP)", "Fluorescein (FITC)", "DAPI"];
%fluorophores = ["DsRed", "EGFP", "SHG", cyan];
%fluorophores = ["iRFP670", "mCherry", "Citrine", "mAzamiGreen", "Cerulean", "eBFP"]
fluorophores = ["Cerulean", "eBFP"]

%Common Channels
%R/fR       [645 685] and [575 630] 
%nR/fR      [634 686] and [573 613]
%CFP/YFP    [520 560] and [460 500]
%Blue/Green [495 540] and [420 460]
%SHG/Green  [495 540] and [370 410]

filter1 = [634 686];
filter2 = [573 613];
filter3 = [520 560];
filter4 = [460 500];

channel_ranges = transpose([filter3; filter4]);
signatures = get_fluorophore_signature(fluorophores, channel_ranges)

channels_to_unmix = [3 5];
mixed_image = mixed_image(:, :, :, channels_to_unmix);
im_size = size(mixed_image);
im_size(end) = num_fluorophores;
unmixed_image = zeros(im_size);

similarities = zeros(num_fluorophores, 1);

max_diff = -10;

for slice = 1:im_size(3)
    for row = 1:im_size(1)
        for col = 1:im_size(2)
            pixel_signature = transpose(squeeze(mixed_image(row, col, slice, :)));
            pixel_intensity = sum(pixel_signature); %Could calculate these some other way
            pixel_signature_normalized = pixel_signature / pixel_intensity;
            for fluorophore = 1:num_fluorophores
                similarities(fluorophore) = mean((pixel_signature_normalized - signatures(fluorophore, :)).^2);
            end
            %similarities = similarities + [0.3; 0];
            sim_diff = similarities(1)-similarities(2);
            if sim_diff>max_diff
               max_diff = sim_diff ;
            end
            if multi_fluorophore_pixels
                inverse_mse = 1./similarities;
                scaled_inverse_mse = inverse_mse/sum(inverse_mse);
                unmixed_image(row, col, slice, :) = pixel_intensity*scaled_inverse_mse;
            else
                [highest_corr, matching_fluorophore] = min(similarities);
                unmixed_image(row, col, slice, matching_fluorophore) = pixel_intensity; 
            end
        end
    end
end



map = zeros(6, 101, 3);
default_map = 0:0.01:1;
map(1, :, 1) = default_map;
map(1, :, 2) = default_map*.2;

map(2, :, 1) = default_map*1;
map(2, :, 2) = default_map*.5;

map(3, :, 1) = default_map*.5;
map(3, :, 2) = default_map*1;

map(4, :, 1) = default_map*.16;
map(4, :, 2) = default_map*1;

map(5, :, 2) = default_map*.92;
map(5, :, 3) = default_map*1;

map(6, :, 2) = default_map*.5;
map(6, :, 3) = default_map*1;



for i = 1:num_fluorophores
    
    %subplot(2, 3, i)
    single_channel_im = unmixed_image(:, :, 1, i);
    if max(single_channel_im(:)) > 0
        figure()
        imshow(single_channel_im, [0 max(single_channel_im(:))])

        colormap(squeeze(map(i, :, :)))
        title(fluorophores(i))
    end
end


save_loc = 'D:\Projects\Channel Unmixing\Results\david_simi_test_stack.tif'; %strcat(fpath, fname);
write_tiff(unmixed_image, save_loc, true, false)
