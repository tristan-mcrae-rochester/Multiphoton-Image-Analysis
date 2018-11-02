clc; clear all; close all;

addpath('D:\Projects\Matlab Tools\Tiff handling');
load_loc = 'D:\Projects\Channel Unmixing\Images\Colorful 800-1050  CFP-YFP cube-crop-high qual-2.tif'; %strcat(fpath, fname); %
num_labels = 3;
num_channels = 5;
all_channel_im = read_tiff(num_channels, load_loc, 10, 10);
all_channel_im = all_channel_im(:, :, :, [3 5]);
dims = size(all_channel_im)
num_channels = dims(end)

num_images = dims(3);
image_size = dims(1:2);
epsilon = eps()*6;
unmixed = zeros(dims(1), dims(2), num_images, num_labels);
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
            Y(ch, low:high) = all_channel_im(:, col+1, slice, ch);
        end
    end
    A0 = transpose([[0.62 0.38]; [0.32 0.68]; [0 1]])
    H0 = rand(num_channels+1, n_pixels); %Y
    [Ac, H] = NNMF(Y, A0, H0, epsilon, max_iter);
    %[Ac, H] = nnmf(Y, num_labels, 'algorithm', 'als', 'w0', A0, 'h0', Y, 'options', opt); 
    Ac = Ac./sum(Ac);
    [vals, indices] = max(Ac, [], 2);
    Ac = Ac(:, indices)
    H = (transpose(Ac) * (Ac+epsilon))\transpose(Ac)*Y;
    for ch = 1:num_labels
        for col = 0:num_cols-1
            low = col*image_size(1)+1;
            high = col*image_size(1)+image_size(1);
            unmixed(:, col+1, slice, ch) = H(ch, low:high);
        end
    end
end

save_loc = 'D:\Projects\Channel Unmixing\Results\cerulean_blue_test.tif'; 
write_tiff(unmixed, save_loc)


