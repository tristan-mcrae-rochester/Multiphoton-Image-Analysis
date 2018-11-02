clc; clear all; %close all;

addpath('D:\Projects\Github\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Local\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'
%'D:\Projects\Local\Channel Unmixing\Images\090618 D7 Rex3 MHC NT-1.tif'
%'D:\Projects\Local\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-higher qual-crop-4.tif'
%'D:\Projects\Local\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-high qual-3.tif'
%'D:\Projects\Local\Channel Unmixing\Images\placenta_location3x3_area3_nonsequential.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif'
%'D:\Projects\Kris & Emma\090618 D7 Rex3 MHC NT single timestep.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif' 
%'D:\Projects\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'

num_fluorophores_to_unmix = 2 
background_as_cluster = true;
extra_clusters=0 %number of additional clusters to use besides the background cluster
num_channels = 3
mixed_image = read_tiff(num_channels, load_loc, 7, 7)*16;
dims = size(mixed_image);
num_rows = dims(1);
num_cols = dims(2);
num_slices = dims(3);
scaled_signatures = false
use_input_intensities = true %Setting this to false is good for debugging where certain clusters are
replicates = 10
multi_fluorophore_pixels = true
pca = false;
background_threshold = 0.00
channels_to_unmix = [2 3]
channels_to_leave = [1]
channels = horzcat(channels_to_unmix,channels_to_leave);

num_fluorophores_to_unmix = num_fluorophores_to_unmix+background_as_cluster+extra_clusters;

if pca
    mixed_image = mixed_image(275:325, 275:325, :, :);
end

pixel_array = image_to_pixel_array(mixed_image);
pixel_sums = mean(pixel_array, 2);
foreground_pixels = pixel_sums > background_threshold;
pixel_array=pixel_array.*foreground_pixels;

pixel_array = pixel_array(:, channels_to_unmix);

%am I scaling pixels correctly?
if scaled_signatures
    if max(pixel_array, [], 2) > 0
        pixel_array = pixel_array./max(pixel_array, [], 2);
    end
end

%PCA for visualization of separation
if pca
    [coeff,score,latent,tsquared,explained,mu] = pca(pixel_array, 'NumComponents', 2);
    biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',{'v_1','v_2','v_3'});
end

[cluster_indices, cluster_centroids, sumd, D] = kmeans(pixel_array, num_fluorophores_to_unmix, 'Replicates', replicates);%, 'Start', initial_centroids);

cluster_centroids

cluster_totals = zeros(num_fluorophores_to_unmix, 1);

for i = 1:num_fluorophores_to_unmix
    cluster_totals(i) = sum(cluster_indices==i);
end
cluster_totals

if multi_fluorophore_pixels
    inverse_square_weights = 1./D.^2;
    inverse_square_weights_pixel_sums = sum(inverse_square_weights, 2);
    cluster_weights = inverse_square_weights./inverse_square_weights_pixel_sums;
else
    %size_D = size(D);
    %cluster_weights = zeros(size_D);
    [D_pixel_mins, pixel_channels] = min(D, [], 2);
    cluster_weights = transpose(ind2vec(transpose(pixel_channels)));
    %cluster_weights(:, pixel_channels) = 1;
    %error("single_fluorophore_pixels is not currently defined")
end


cluster_weights = pixel_array_to_image(cluster_weights, num_slices, num_rows, num_cols);


%cluster_weights for each pixel sum to the total intensity of the channels
%being unmixed after this step
if use_input_intensities
   cluster_weights = cluster_weights .*sum(mixed_image(:, :, :, channels_to_unmix), 4); 
end

dims(end) = num_fluorophores_to_unmix+length(channels_to_leave);
unmixed_image = zeros(dims);

if length(channels_to_leave)>0
    unmixed_image(:, :, :, 1:length(channels_to_leave)) = mixed_image(:, :, :, channels_to_leave);
end
unmixed_image(:, :, :, length(channels_to_leave)+1:end) = cluster_weights;






unmixed_image = unmixed_image/max(unmixed_image(:));

save_loc = 'D:\Projects\Local\Channel Unmixing\Results\k_means_example.tif'; %strcat(fpath, fname);
write_tiff(unmixed_image, save_loc, true, false)





