clc; clear all; %close all;

addpath('D:\Projects\Github\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Local\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'
%'D:\Projects\Local\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-higher qual-crop-4.tif'
%'D:\Projects\Local\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-high qual-3.tif'
%'D:\Projects\Local\Channel Unmixing\Images\placenta_location3x3_area3_nonsequential.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif'
%'D:\Projects\Kris & Emma\090618 D7 Rex3 MHC NT single timestep.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif' 
%'D:\Projects\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'


num_channels = 3
num_clusters = 3
mixed_image = read_tiff(num_channels, load_loc, 10, 10)*16;
mixed_image = mixed_image(275:325, 275:325, :, :);
pixel_array = image_to_pixel_array(mixed_image);

T = clusterdata(pixel_array, num_clusters);

unmixed_image = pixel_array_to_image(T, 1, 50, 50);