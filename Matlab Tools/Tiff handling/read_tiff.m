function image = read_tiff(num_channels, fname, start_slice, stop_slice)

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
        start_slice = 1;
        stop_slice = num_images/num_channels;
    case 3
        stop_slice = num_images/num_channels;
end

%Initialize image with zeros
image = zeros([image_size, stop_slice-start_slice+1, num_channels]);


%Read tiff into array
for slice_index = start_slice:stop_slice
    for channel = 1:num_channels
        slice = im2double(imread(fname, (slice_index-1)*num_channels+channel));
        image(:, :, slice_index-start_slice+1, channel) = slice;
        %max(slice(:))
    end
end













