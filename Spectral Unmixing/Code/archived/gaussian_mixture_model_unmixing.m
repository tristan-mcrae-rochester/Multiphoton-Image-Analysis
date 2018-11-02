clc; clear all; close all;

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
num_clusters = 4 %Number of fluorophores to unmix + 1 for hte backlground at minimum
use_input_intensities = false %setting this to false is good for visualizing clusters during debugging
scaled_signatures = false %setting this to true makes it not converge
background_threshold = 0.01
mixed_image = read_tiff(num_channels, load_loc, 10, 10)*16;
dims = size(mixed_image);
num_rows = dims(1);
num_cols = dims(2);
num_slices = dims(3);

pixel_array = image_to_pixel_array(mixed_image);

pixel_sums = mean(pixel_array, 2);
foreground_pixels = pixel_sums > background_threshold;
pixel_array=pixel_array.*foreground_pixels;

if scaled_signatures
    if max(pixel_array, [], 2) > 0
        pixel_array = pixel_array./max(pixel_array, [], 2);
    end
end


gm = fitgmdist(pixel_array,num_clusters, 'SharedCov', true);%, 'CovType', 'diagonal' %This doesn't always converge and throws an error when it doesn't

cluster_indices = cluster(gm, pixel_array);

split_image = pixel_array_to_image(cluster_indices, num_slices, num_rows, num_cols);

dims(end) = num_clusters;
unmixed_image = zeros(dims);

for i = 1:num_clusters
    
    cluster_totals(i) = sum(cluster_indices==i);
    cluster_image = (split_image==i);
    if use_input_intensities
       cluster_image = cluster_image .*sum(mixed_image, 4); 
    end
    unmixed_image(:, :, :, i) = cluster_image;
    subplot(1, num_clusters, i)
    imshow(squeeze(unmixed_image(:,:,1,i)))
end

gm
