clc; clear all; close all;

addpath('D:\Projects\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Kris & Emma\090618 D7 Rex3 MHC NT single timestep.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif' 
num_fluorophores = 4;
num_channels = 4;
multi_fluorophore_pixels = false;
mixed_image = read_tiff(num_channels, load_loc, 20, 20)*16;
MaiTai = true;
InSight = true;
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
%fluorophores = ["Cerulean", "eBFP"]
fluorophores  = ["iRFP670", "eGFP", "SHG", "eBFP"]

%Common Channels
%R/fR       [645 685] and [575 630] 
%nR/fR      [634 686] and [573 613]
%CFP/YFP    [520 560] and [460 500]
%Blue/Green [495 540] and [420 460]
%SHG/Green  [495 540] and [370 410]

filter1 = [645 685];
filter2 = [575 630];
filter3 = [495 540];
filter4 = [420 460];

im_size = size(mixed_image);
im_size(end) = num_fluorophores;
unmixed_image = zeros(im_size);

%Pick up fluorophores excited by MaiTai
if MaiTai
    channel_ranges = transpose([filter3; filter4]);
    fluorophore_indices = [4]
    signatures = get_fluorophore_signature(fluorophores(fluorophore_indices), channel_ranges)
    channels_to_unmix = [4];
    mixed_image_laser_0 = mixed_image(:, :, :, channels_to_unmix);
    unmixed_image = get_fluorophore_from_channel(im_size, fluorophore_indices, signatures, mixed_image_laser_0, unmixed_image, multi_fluorophore_pixels);
end

%Pick up fluorophores excited by insight
if InSight
    channel_ranges = transpose([filter1; filter2; filter3]);
    fluorophore_indices = [1 2 3]
    signatures = get_fluorophore_signature(fluorophores(fluorophore_indices), channel_ranges)
    channels_to_unmix = [1 2 3];
    mixed_image_laser_1 = mixed_image(:, :, :, channels_to_unmix);
    unmixed_image = get_fluorophore_from_channel(im_size, fluorophore_indices, signatures, mixed_image_laser_1, unmixed_image, multi_fluorophore_pixels);
end


% map = zeros(6, 101, 3);
% default_map = 0:0.01:1;
% map(1, :, 1) = default_map;
% map(1, :, 2) = default_map*.2;
% map(2, :, 1) = default_map*1;
% map(2, :, 2) = default_map*.5;
% map(3, :, 1) = default_map*.5;
% map(3, :, 2) = default_map*1;
% map(4, :, 1) = default_map*.16;
% map(4, :, 2) = default_map*1;
% map(5, :, 2) = default_map*.92;
% map(5, :, 3) = default_map*1;
% map(6, :, 2) = default_map*.5;
% map(6, :, 3) = default_map*1;
% for i = 1:num_fluorophores
%     
%     %subplot(2, 3, i)
%     single_channel_im = unmixed_image(:, :, 1, i);
%     if max(single_channel_im(:)) > 0
%         figure()
%         imshow(single_channel_im, [0 max(single_channel_im(:))])
% 
%         colormap(squeeze(map(i, :, :)))
%         title(fluorophores(i))
%     end
% end


save_loc = 'D:\Projects\Kris & Emma\Results\simi_test_stack.tif'; %strcat(fpath, fname);
write_tiff(unmixed_image, save_loc, true, false)
