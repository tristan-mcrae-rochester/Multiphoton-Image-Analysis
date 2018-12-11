
function unmixed_image = k_means_unmixing(num_channels, num_timesteps, num_fluorophores_to_unmix, channels_to_unmix, image_path, save_type)

    mixed_image = read_image_from_path(image_path, num_channels, num_timesteps);

    background_as_cluster = true;
    dims = size(mixed_image);
    x_dim = dims(1);
    y_dim = dims(2);
    z_dim = dims(3);
    scaled_signatures = true
    use_input_intensities = true; %Setting this to false is good for debugging where certain clusters are
    replicates = 1
    multi_fluorophore_pixels = true
    background_threshold = 0.00;
    maximum_iterations = 100;
    channels = 1:num_channels;
    channels_to_leave = channels(~ismember(channels, channels_to_unmix));
    num_fluorophores_to_unmix = num_fluorophores_to_unmix+background_as_cluster;

    
    disp(num_timesteps)
    disp(size(mixed_image))
    
    %It doesn't like doing kmeans on unsigned integers (understandibly)
    mixed_image = double(mixed_image);
    unscaled_mixed_image = mixed_image;
    
    %Scale pixels for clustering only
    if scaled_signatures
        disp("Scaling pixel intensities")
        mixed_image(:, :, :, channels_to_unmix, :) = mixed_image(:, :, :, channels_to_unmix, :)/max(mixed_image(:)); %./max(mixed_image(:, :, :, channels_to_unmix, :), [], 4) %change to just be for channels we're unmixing
    end
    
    representitive_timestep = int32(num_timesteps/2)
    %Puts channels in first axis so they can be saved for k-means
    if num_timesteps>1
        reordered_mixed_image = permute(mixed_image, [4 1 2 3 5]);
        reordered_mixed_image = reordered_mixed_image(:, :, :, :, representitive_timestep);
    else
        reordered_mixed_image = permute(mixed_image, [4 1 2 3]);
    end
    
    
    
    pixel_array = reordered_mixed_image(:, :);%image_to_pixel_array(mixed_image);
    pixel_array = permute(pixel_array, [2 1]);

    %Filter out background pixels
    pixel_sums = mean(pixel_array, 2);
    foreground_pixels = pixel_sums > background_threshold;
    pixel_array(~foreground_pixels)=0;

    pixel_array = pixel_array(:, channels_to_unmix); 

    
      
    close all;    
    disp('Running K-Means. The display may freeze but it is still working behind the scenes. This may take a few minutes for large datasets.')
    [cluster_indices, cluster_centroids, sumd, D] = kmeans(pixel_array, num_fluorophores_to_unmix, 'Replicates', replicates, 'MaxIter', maximum_iterations, 'Display', 'iter');
    
    disp('Finished Running k-means')
    pause(0.01) %let the display catch up
    
    
    cluster_centroids

    
    cluster_totals = zeros(num_fluorophores_to_unmix, 1);

    for i = 1:num_fluorophores_to_unmix
        cluster_totals(i) = sum(cluster_indices==i);
    end
    cluster_totals;

    
    num_output_channels = num_fluorophores_to_unmix+length(channels_to_leave);
    cluster_weights_image = zeros(x_dim, y_dim, z_dim, num_fluorophores_to_unmix, num_timesteps);
    cluster_weights_timestep = zeros(x_dim, y_dim, z_dim, num_fluorophores_to_unmix);
    distances = zeros(x_dim, y_dim, z_dim, num_fluorophores_to_unmix);
    cluster_center_broadcast = zeros(x_dim, y_dim, z_dim, length(channels_to_unmix), num_fluorophores_to_unmix);
    
    disp('Broadcasting cluster centers')
    for cluster = 1:num_fluorophores_to_unmix
        cluster_center = cluster_centroids(cluster, :);
        for x = 1:x_dim
            for y = 1:y_dim
                for z = 1:z_dim
                    cluster_center_broadcast(x, y, z, :, cluster) = cluster_center;    
                end
            end
        end
    end
    
    for t = 1:num_timesteps
        disp(strcat("applying clustering to timestep ", int2str(t), " of ", int2str(num_timesteps)))
        if multi_fluorophore_pixels
            if num_timesteps == 1
                inverse_square_weights = 1./D.^2;
                inverse_square_weights_pixel_sums = sum(inverse_square_weights, 2);
                cluster_weights = inverse_square_weights./inverse_square_weights_pixel_sums;
                
                cluster_weights_image(:) = cluster_weights(:);
            else
                for cluster = 1:num_fluorophores_to_unmix   
                    pixel_values = double(mixed_image(:, :, :, channels_to_unmix, t));
                    diff = pixel_values - cluster_center_broadcast(:, :, :, :, cluster);
                    intermediate_step = vecnorm(diff, 2, 4); 
                    distances(:, :, :, cluster) = squeeze(intermediate_step);
                end
                pause(0.01)
                inverse_square_weights = 1./distances.^2;
                inverse_square_weights_pixel_sums = sum(inverse_square_weights, 4);
                cluster_weights = inverse_square_weights./inverse_square_weights_pixel_sums;
                cluster_weights_timestep(:) = cluster_weights(:);
                cluster_weights_image(:, :, :, :, t) = cluster_weights_timestep;
            end
        else
            if num_timesteps == 1
                [D_pixel_mins, pixel_channels] = min(D, [], 2);
                cluster_weights = transpose(ind2vec(transpose(pixel_channels)));
                cluster_weights_image(:) = cluster_weights(:);
            else
                disp('Functionality for multiple timesteps with one fluorophore per pixel is not available')
            end
        end
    end
 

    %cluster_weights for each pixel sum to the total intensity of the channels
    %being unmixed after this step
    if use_input_intensities
        %Is this part right?
        cluster_weights_image = cluster_weights_image .*mean(unscaled_mixed_image(:, :, :, channels_to_unmix, :), 4); 
    else
        cluster_weights_image = cluster_weights_image * double(max(unscaled_mixed_image(:)));
    end

    unmixed_image = zeros(x_dim, y_dim, z_dim, num_output_channels, num_timesteps);

    if length(channels_to_leave)>0
        unmixed_image(:, :, :, 1:length(channels_to_leave), :) = mixed_image(:, :, :, channels_to_leave, :);
    end
    unmixed_image(:, :, :, length(channels_to_leave)+1:end, :) = cluster_weights_image;
    
    unmixed_image = unmixed_image/max(unscaled_mixed_image(:));
    
    write_tiff(unmixed_image, 'test_multi_tiff_save', save_type); %4d single tiff test save.tif

end 


function mixed_image = read_image_from_path(image_path, num_channels, num_timesteps)
    image_path_split = strsplit(image_path, '.')
    extension = image_path_split(2)
    
    if extension == "tiff" || extension == "tif"
        mixed_image = read_tiff(num_channels, image_path, num_timesteps)*16;
    elseif extension == "oir"
        mixed_image = read_oir(image_path, num_channels, num_timesteps);
    elseif extension == "oif"
        mixed_image = read_oif(image_path, num_channels, num_timesteps);
    else
        disp(strcat(extension, " is an unrecognized file type"))
    end    

end

function mixed_image = read_oif(image_path, num_channels, num_timesteps)
    data = bfopen(image_path);
    [num_images, unused] = size(data{1, 1});
    [n_rows, n_cols] = size(data{1, 1}{1, 1});
    num_slices = (num_images/num_channels)/num_timesteps;
    mixed_image = zeros(n_rows, n_cols, num_slices, num_channels, num_timesteps); 
    
    for t = 1:num_timesteps
        for z = 1:num_slices
           for c = 1:num_channels
               mixed_image(:, :, z, c, t) = data{1, 1}{(t-1)*num_channels*num_slices+(z-1)*num_channels+c, 1};
           end
       end
    end
end

function mixed_image = read_oir(image_path, num_channels, num_timesteps)
    data = bfopen(image_path);
    [num_images, unused] = size(data{1, 1});
    [n_rows, n_cols] = size(data{1, 1}{1, 1});
    num_slices = (num_images/num_channels)/num_timesteps;
    mixed_image = zeros(n_rows, n_cols, num_slices, num_channels, num_timesteps);
    for z = 1:num_slices
       for t = 1:num_timesteps
           for c = 1:num_channels
               mixed_image(:, :, z, c, t) = data{1, 1}{(z-1)*num_channels*num_timesteps+(t-1)*num_channels+c, 1};
           end
       end
    end
end

function mixed_image = read_tiff(num_channels, fname, num_timesteps, start_slice, stop_slice)
    %Prompt user for file path if none given
    if nargin == 1
        [fname, fpath] =  uiputfile;
            fname = strcat(fpath, fname);
    end

    %Gather basic information about tiff being read
    info = imfinfo(fname);
    num_images = numel(info);
    image_size = size(imread(fname));

    %Set default slices to read
    switch nargin
        case 2
            num_timesteps = 1;
            start_slice = 1;
            stop_slice = num_images/num_channels/num_timesteps;
            
        case 3
            start_slice = 1;
            stop_slice = num_images/num_channels/num_timesteps;
    end

    if length(image_size) == 3 & num_channels>1
        mixed_image = im2double(imread(fname));
        return
    end
    %Initialize image with zeros
    mixed_image = zeros([image_size, stop_slice-start_slice+1, num_channels, num_timesteps]);

    %Read tiff into array
    for slice_index = start_slice:stop_slice
        for t = 1:num_timesteps
            for channel = 1:num_channels
                slice = im2double(imread(fname, (t-1)*(stop_slice-start_slice)*(num_channels)+(slice_index-1)*num_channels+channel));
                mixed_image(:, :, slice_index-start_slice+1, channel, t) = slice;
            end
        end
    end
end


function write_tiff(image, fname, save_type)
    dims = size(image);
    
    if length(dims) >=3
        num_slices = dims(3)
    else
        num_slices = 1  
    end
    if length(dims) >= 4
        num_channels = dims(4)
    else
        num_channels = 1   
    end
    if length(dims) >= 5
        num_timesteps = dims(5)
    else
        num_timesteps = 1
    end
        
    
    if save_type == "single_tiff"
        delete(fname)
        for t = 1:num_timesteps
            for z = 1:num_slices
                for c = 1:num_channels
                    imwrite(double(squeeze(image(:, :, z, c, t))), fname, 'WriteMode', 'append', 'Compression', 'none');
                end
            end
        end
    elseif save_type == "multiple_tiffs"
        mkdir(fname)
        for t = 1:num_timesteps
            for z = 1:num_slices
                for c = 1:num_channels
                    fname_2d = [fname, '\s_C', int2str(c), 'Z', int2str(z), 'T', int2str(t), '.tif'];
                    delete(fname_2d)
                    imwrite(double(squeeze(image(:, :, z, c, t))), fname_2d, 'WriteMode', 'append', 'Compression', 'none');
                end
            end
        end
    end
end



