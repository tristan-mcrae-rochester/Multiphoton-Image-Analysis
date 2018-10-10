clc; clear all; %close all;

addpath('D:\Projects\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif'
%'D:\Projects\Kris & Emma\090618 D7 Rex3 MHC NT single timestep.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 840-1050 BFP-GFP cube-higher qual-crop-3.tif' 
%'D:\Projects\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'

num_fluorophores = 6;
num_channels = 5;
mixed_image = read_tiff(num_channels, load_loc, 20, 20)*16;
dims = size(mixed_image);
num_rows = dims(1);
num_cols = dims(2);
num_slices = dims(3);
scaled_signatures = true;
use_input_intensities = true;
replicates = 10;
multi_fluorophore_pixels = true;
median_filter = false;
pca = false;

num_fluorophores = num_fluorophores+1;

if pca
    mixed_image = mixed_image(275:325, 275:325, :, :);
end

pixel_array = image_to_pixel_array(mixed_image);
if scaled_signatures
    pixel_array = pixel_array./max(pixel_array, [], 2);
end

%PCA for visualization of separation
if pca
    [coeff,score,latent,tsquared,explained,mu] = pca(pixel_array, 'NumComponents', 2);
    biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',{'v_1','v_2','v_3'});
end

[cluster_indices, cluster_centroids, sumd, D] = kmeans(pixel_array, num_fluorophores, 'Replicates', replicates);%, 'Start', initial_centroids);


cluster_centroids
%max(D(:))



split_image = pixel_array_to_image(cluster_indices, num_slices, num_rows, num_cols);

cluster_totals = zeros(num_fluorophores, 1);
dims(end) = num_fluorophores;
unmixed_image = zeros(dims);

for i = 1:num_fluorophores
    cluster_totals(i) = sum(cluster_indices==i);
    %subplot(3, 3, i)
    cluster_image = (split_image==i);
    if use_input_intensities
       cluster_image = cluster_image .*sum(mixed_image, 4); 
    end
    %imshow(cluster_image)
    if median_filter
       cluster_image = medfilt2(cluster_image); 
    end
    unmixed_image(:, :, :, i) = cluster_image;
end
cluster_totals


if multi_fluorophore_pixels
    inverse_square_weights = 1./D.^2;
    inverse_square_weights_pixel_sums = sum(inverse_square_weights, 2);
    cluster_weights = inverse_square_weights./inverse_square_weights_pixel_sums;
    %weights = ((2-D)./2).^20;
    cluster_weights = pixel_array_to_image(cluster_weights, num_slices, num_rows, num_cols);
    unmixed_image = cluster_weights;
    if use_input_intensities
       unmixed_image = unmixed_image .*sum(mixed_image, 4); 
    end
    mean(cluster_weights(:))
    D(1, :)
    transpose(squeeze(cluster_weights(1, 1, 1, :)))
end

unmixed_image = unmixed_image/max(unmixed_image(:));

save_loc = 'D:\Projects\Channel Unmixing\Results\k_means_test_stack.tif'; %strcat(fpath, fname);
write_tiff(unmixed_image, save_loc, true, false)





