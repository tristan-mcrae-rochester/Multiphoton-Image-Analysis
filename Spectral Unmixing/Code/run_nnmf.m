clc; clear all; close all;

addpath('D:\Projects\Github\Matlab Tools\Tiff handling');
%[fname, fpath] =  uigetfile;
load_loc = 'D:\Projects\Local\Channel Unmixing\Images\BPAE cells nonsequential area2.tif'
%'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'; %strcat(fpath, fname); %
%data = bfopen(load_loc);


num_labels=3;
num_channels = 3;%input('How many image channels are there? \n'); %3;%
all_channel_im = read_tiff(num_channels, load_loc, 7, 7)*16;


% for i = 1:4
%     slice = im2double(imread(load_loc, i));
%     all_channel_im(:, :, 1, i) = slice;
% end

%all_channel_im = all_channel_im(:, :, :, 2:4);

dims = size(all_channel_im)
num_images = dims(3);%/num_channels;
image_size = dims(1:2);
epsilon = eps()*6;
unmixed = zeros(dims(1), dims(2), 1, num_labels);  %zeros(dims);
num_cols = image_size(2);
autofluorescence = false;
max_iter = 150;
opt = statset('MaxIter', max_iter);
n_pixels = prod(image_size);

for slice = 1:num_images
    
    Y  = zeros(num_channels+autofluorescence, n_pixels);
    for ch = 1:num_channels
        for col = 0:num_cols-1
            low = col*image_size(1)+1;
            high = col*image_size(1)+image_size(1);
            Y(ch, low:high) = all_channel_im(:, col+1, slice, ch);
        end
    end
    
    
    if autofluorescence
%         A0 = transpose([[1 0 0]; [.3511 .6489 0]; [.108 .5023 .3896]; [1/3 1/3 1/3]]);
%         H0 = rand(num_channels+autofluorescence, n_pixels);
%         [Ac, H] = NNMF(Y, A0, H0, epsilon, max_iter);
        A0 = transpose([[.8 .2 0 0]; [.3511 .6489 0 0]; [.108 .5023 .3896 0]; [0 0 0 0]]);
        [Ac, H] = nnmf(Y, num_channels+autofluorescence, 'algorithm', 'als', 'w0', A0, 'h0', Y, 'options', opt);
        
    else
        A0 = transpose([[1 0 0]; [.3511 .6489 0]; [.108 .5023 .3896]]);
        %A0 = transpose([[1 0 0 0]; [.2 .6 0 .2]; [.1 0 1 0]; [0 0 0 1]]);
        %A0 = transpose([[.2 .4 .4 0]; [0 .1 .4 .5]])
        %A0 = transpose([[1 0 0]; [.5 .5 0]; [.1 .4 .5]])
        %[Ac, H] = NNMF(Y, A0, Y, epsilon, max_iter);
        [Ac, H] = nnmf(Y, num_labels, 'algorithm', 'als', 'w0', A0, 'h0', Y, 'options', opt); %'w0', A0, 'h0', Y,
    end
    
    
    Ac = Ac./sum(Ac);
    [vals, indices] = max(Ac, [], 2);
    Ac = Ac(:, indices)
    H = (transpose(Ac) * (Ac+epsilon))\transpose(Ac)*Y;
    
    for ch = 1:num_labels
        for col = 0:num_cols-1
            low = col*image_size(1)+1;
            high = col*image_size(1)+image_size(1);
            unmixed(:, col+1, 1, ch) = H(ch, low:high); %change "1" in 3rd entry to "slice" when running on multiple slices
        end
    end
end

%[fname, fpath] =  uiputfile;
save_loc = 'D:\Projects\Local\Channel Unmixing\Results\nnmf_example.tif'; 
write_tiff(unmixed, save_loc)


