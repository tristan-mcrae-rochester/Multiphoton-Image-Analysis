%
%
%  K-means Spectral Clustering Function for Imaris 
%
%  Tristan McRae 2018
%  References - http://www.scs2.net/next/files/courses/iic/ImarisOpenLaunchExtract.pdf
%
%  Installation:
%
%  - Copy this file into the XTensions folder in the Imaris installation directory
%  - You will find this function in the Image Processing menu
%
%    <CustomTools>
%      <Menu>
%        <Submenu name="Custom Extensions">
%          <Item name="Spectral Unmixing" icon="Matlab" tooltip="Split overlapping fluorophores to each have their own channel">
%            <Command>MatlabXT::XTSpectralUnmixing(%i)</Command>
%          </Item>
%        </Submenu>
%      </Menu>
%    </CustomTools>
%  
%
%  Description:
%
%   This XTension separates pixels into clusters based on their spectral
%   signatures
%   This helps to discriminate between pixels that share a dominant
%   channel but belong to different parts of the cell
%
%



function XTSpectralUnmixing(aImarisApplicationID)

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Load data
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ID
disp('Loading data into MATLAB (for large datasets, this may take some time)')

disp('Setting up connection')
% Set up connection	between	Imaris and MATLAB
if	isa(aImarisApplicationID, 'Imaris.IApplicationPrxHelper')
    vImarisApplication = aImarisApplicationID;
else
    % connect to Imaris interface
    javaaddpath ImarisLib.jar
    vImarisLib = ImarisLib;
    if	ischar(aImarisApplicationID)
        aImarisApplicationID = round(str2double(aImarisApplicationID));
    end
    vImarisApplication = vImarisLib.GetApplication(aImarisApplicationID);
end

disp('Finding the dataset properties')
% Get the vDataSet object
vDataSet = vImarisApplication.GetDataSet();
% Get the data type
switch char(vDataSet.GetType())
    case 'eTypeUInt8', datatype = 'uint8';
    case 'eTypeUInt16', datatype = 'uint16';
    case 'eTypeFloat', datatype = 'single';
    otherwise, error('Bad value for vDataSet.GetType()');
end
% Allocate memory
disp('Allocating Memory')
x_dim = vDataSet.GetSizeX();
y_dim = vDataSet.GetSizeY();
z_dim = vDataSet.GetSizeZ();
num_channels = vDataSet.GetSizeC();
num_timesteps = vDataSet.GetSizeT();

stack = zeros([x_dim, y_dim, z_dim, num_channels, num_timesteps], datatype);
basic_stack = zeros([x_dim, y_dim, z_dim], datatype);
% Get the pixel data
for t = 1:num_timesteps
    disp(strcat("Reading timestep ", int2str(t), " of ", int2str(num_timesteps)))
    pause(0.01) %let the display catch up
    for c = 1:num_channels
        switch char(vDataSet.GetType())
            case 'eTypeUInt8'
                arr = vDataSet.GetDataVolumeAs1DArrayBytes(c-1, t-1); 
                basic_stack(:) = typecast(arr, 'uint8');
                stack(:, :, :, c, t) = basic_stack;
                
            case 'eTypeUInt16'
                arr	= vDataSet.GetDataVolumeAs1DArrayShorts(c-1, t-1);
                basic_stack(:) = typecast(arr, 'uint16');
                stack(:, :, :, c, t) = basic_stack;
                
            case 'eTypeFloat'
                basic_stack(:) = vDataSet.GetDataVolumeAs1DArrayFloats(c-1, t-1);
                stack(:, :, :, c, t) = basic_stack; 
            otherwise, error('Bad value for type');
        end
    end
end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

unmixed_image  = spectral_unmixing_gui(num_channels, num_timesteps, stack);
disp('returning image')
return_results_to_imaris(unmixed_image, vDataSet, num_timesteps, vImarisApplication);

end

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%Prompt user for unmixing details
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function unmixed_image = spectral_unmixing_gui(num_channels, num_timesteps, mixed_image)
    close all;

    h.f = figure('units','pixels','position',[700,500,150,100],...
                 'toolbar','none','menu','none', 'Name', 'Unmixing Method', 'NumberTitle', 'off');

    h.t = uicontrol('style', 'text','units', 'pixels',...
        'position',[0, 60, 150, 30], 'string', 'Select an unmixing method');           

    h.d = uicontrol('style', 'popupmenu', 'units', 'pixels',...
        'position',[10, 30, 90, 30], 'string', 'K-means|SIMI (coming soon!)|NNMF (coming soon!)');

    h.p = uicontrol('style','pushbutton','units','pixels',...
                    'position',[40,5,70,20],'string','OK',...
                    'callback',@d_call);
    uiwait(h.f)
    % Dropdown callback
    function d_call(varargin)
        vals = get(h.d,'Value');
        switch vals
            case 1
                unmixed_image = k_means_gui(num_channels, num_timesteps, mixed_image); 
            case 2
                unmixed_image = simi_gui(mixed_image); 
            case 3
                unmixed_image = nnmf_gui(mixed_image); 

        end
        close all;
    end

end


function unmixed_image = k_means_gui(num_channels, num_timesteps, mixed_image)
    h.f = figure('units','pixels','position',[700,500,max(20+40*num_channels, 300),130],...
                 'toolbar','none','menu','none', 'Name', 'K-Means Settings', 'NumberTitle', 'off');

    h.t(1) = uicontrol('style', 'text','units', 'pixels',...
        'position',[0, 90, 300, 30], 'string', 'How many fluorophores do you want to unmix?');      

    h.s = uicontrol('style', 'edit','units', 'pixels',...
        'position',[20, 70, 40, 30]);        

    h.t(2) = uicontrol('style', 'text','units', 'pixels',...
        'position',[0, 40, 200, 30], 'string', 'Select which channels to unmix');

    for i = 1:num_channels
        h.c(i) = uicontrol('style','checkbox','units','pixels',...
                        'position',[10+40*(i-1),30,50,15],'string',int2str(i));
    end

    h.p = uicontrol('style','pushbutton','units','pixels',...
                    'position',[40,5,70,20],'string','OK',...
                    'callback',@p_call);
    uiwait(h.f)

    function p_call(varargin)
        vals = get(h.c,'Value');
        if length(vals)>1
            channels_to_unmix = find([vals{:}]);
        else
            channels_to_unmix = find([vals]);
        end
        num_fluorophores_to_unmix = str2num(get(h.s, 'String'));

        unmixed_image = k_means_unmixing(num_channels, num_timesteps, num_fluorophores_to_unmix, channels_to_unmix, mixed_image);
        uiresume
    end
end

function unmixed_image = simi_gui()

end

function unmixed_image = nnmf_gui(mixed_image)
    unmixed_image = nnmf_unmixing(mixed_image);
end


function unmixed_image = k_means_unmixing(num_channels, num_timesteps, num_fluorophores_to_unmix, channels_to_unmix, mixed_image)

    background_as_cluster = true;
    dims = size(mixed_image);
    x_dim = dims(1);
    y_dim = dims(2);
    z_dim = dims(3);
    scaled_signatures = true
    use_input_intensities = true; %Setting this to false is good for debugging where certain clusters are
    replicates = 5
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
        mixed_image(:, :, :, channels_to_unmix, :) = mixed_image(:, :, :, channels_to_unmix, :)./max(mixed_image(:, :, :, channels_to_unmix, :), [], 4); %change to just be for channels we're unmixing
    end
    
    representitive_timestep = int32(num_timesteps/2)
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
    [cluster_indices, cluster_centroids, sumd, D] = kmeans(pixel_array, num_fluorophores_to_unmix, 'Replicates', replicates, 'MaxIter', maximum_iterations, 'Display', 'iter');%, 'Start', initial_centroids);
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
        cluster_weights_image = cluster_weights_image .*mean(unscaled_mixed_image(:, :, :, channels_to_unmix, :), 4); 
    else
        cluster_weights_image = cluster_weights_image * double(max(unscaled_mixed_image(:)));
    end

    unmixed_image = zeros(x_dim, y_dim, z_dim, num_output_channels, num_timesteps);

    if length(channels_to_leave)>0
        %disp('Change this when running for multiple timesteps')
        unmixed_image(:, :, :, 1:length(channels_to_leave), :) = mixed_image(:, :, :, channels_to_leave, :);
    end
    unmixed_image(:, :, :, length(channels_to_leave)+1:end, :) = cluster_weights_image;

end %breakpoint

function unmixed_image = nnmf_unmixing(mixed_image)
    disp('starting nnmf unmixing')
    %This can't currently handle 4D data
    %This assumes # of input channels is also # of output channels because
    %that's all I've seen it do correctly


    %I think this has issues with the scale and datatype of the incoming
    %data similar to what k-means had at first
    
    disp(max(mixed_image(:)))
    mixed_image = double(mixed_image)./double(max(mixed_image(:)));
    disp(max(mixed_image(:)))
    
    dims = size(mixed_image);
    num_images = dims(3);
    num_channels = dims(4);
    num_labels = num_channels; %In naive case
    image_size = dims(1:2);
    epsilon = eps()*6;
    unmixed_image = zeros(dims);  
    num_cols = image_size(2);
    max_iter = 150;
    opt = statset('MaxIter', max_iter);
    n_pixels = prod(image_size);

    for slice = 1:num_images

        Y  = zeros(num_channels, n_pixels);
        for ch = 1:num_channels
            for col = 0:num_cols-1
                low = col*image_size(1)+1;
                high = col*image_size(1)+image_size(1);
                Y(ch, low:high) = mixed_image(:, col+1, slice, ch);
            end
        end


        A0 = transpose([[1 0 0]; [.3511 .6489 0]; [.108 .5023 .3896]]);
        [Ac, H] = nnmf(Y, num_labels, 'algorithm', 'als', 'w0', A0, 'h0', Y, 'options', opt); 


        Ac = Ac./sum(Ac);
        [vals, indices] = max(Ac, [], 2);
        Ac = Ac(:, indices);
        H = (transpose(Ac) * (Ac+epsilon))\transpose(Ac)*Y;

        for ch = 1:num_labels
            for col = 0:num_cols-1
                low = col*image_size(1)+1;
                high = col*image_size(1)+image_size(1);
                unmixed(:, col+1,slice, ch) = H(ch, low:high); 
            end
        end
    end


end

function return_results_to_imaris(unmixed_image, vDataSet, num_timesteps, vImarisApplication)

% Create a new channel
dims = size(unmixed_image);
num_output_channels = dims(4);
vDataSet.SetSizeC(num_output_channels);            
%num_timesteps = 1 %For debugging only

for t = 1:num_timesteps
    disp(strcat("returning timestep ", int2str(t), " of ", int2str(num_timesteps)))
    pause(0.01)
    for c = 1:num_output_channels
        basic_stack = unmixed_image(:, :, :, c, t);
        switch char(vDataSet.GetType())
            case 'eTypeUInt8'
                vDataSet.SetDataVolumeAs1DArrayBytes(uint8(basic_stack(:)), c-1, t-1); 
            case 'eTypeUInt16'
                vDataSet.SetDataVolumeAs1DArrayShorts(uint16(basic_stack(:)), c-1, t-1);
            case 'eTypeFloat'
                vDataSet.SetDataVolumeAs1DArrayFloats(single(basic_stack(:)), c-1, t-1);
            otherwise, error('Bad value for type');
        end
    end
end

vImarisApplication.SetDataSet(vDataSet);


end
