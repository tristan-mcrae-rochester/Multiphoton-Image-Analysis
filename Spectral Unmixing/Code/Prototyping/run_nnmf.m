clc; clear all; close all;

%[fname, fpath] =  uigetfile; % 'BPAE cells nonsequential.tif' ;
%load_loc = strcat(fpath, fname);
load_loc = 'C:\Users\tmcrae\Desktop\Projects\Channel_Unmixing\Images\BPAE cells nonsequential.tif';
num_channels = 3;
all_channel_im = read_tiff(load_loc, num_channels);
dims = size(all_channel_im);
num_images = dims(3)/num_channels;
my_scale = dims(1:2);
epsilon = eps()*6;
unmixed = zeros(dims);
show_slice = 9;
num_cols = my_scale(2);

for slice = 1:num_images
    
    Y  = zeros(num_channels, prod(my_scale));%Observation matrix
    for ch = 1:num_channels
        for col = 0:num_cols-1
            low = col*my_scale(1)+1;
            high = col*my_scale(1)+my_scale(1);
            Y(ch, low:high) = all_channel_im(:, col+1, slice, ch);
        end
    end
    
    A0 = transpose([[1 0 0]; [.3511 .6489 0]; [.108 .5023 .3896]]);
    %[Ac, H] = NNMF(Y, A0, epsilon);
    opt = statset('MaxIter', 200);
    [Ac, H] = nnmf(Y, num_channels, 'algorithm', 'als', 'w0', A0, 'h0', Y, 'options', opt); %'mult' is inferior to 'als'?
    Ac = Ac./sum(Ac);
    [vals, indices] = max(Ac, [], 2);
    Ac = Ac(:, indices);
    H = (transpose(Ac) * (Ac+epsilon))\transpose(Ac)*Y;
    
    for ch = 1:num_channels
        for col = 0:num_cols-1
            low = col*my_scale(1)+1;
            high = col*my_scale(1)+my_scale(1);
            unmixed(:, col+1, slice, ch) = H(ch, low:high);
        end
    end
    

%     figure
%     subplot(121)
%     imshow(squeeze(all_channel_im(:, :, slice, :))*16)
%     title('Non Sequential Raw')
%     subplot(122)
%     imshow(squeeze(unmixed(:, :, slice, :))*16)
%     title('Non Sequential NNMF Unmixed')
    
    
    
end


[fname, fpath] =  uiputfile;
save_loc = strcat(fpath, fname);
delete(save_loc)
write_tiff(unmixed, save_loc)


% 
% microscope_unmixed    = read_tiff('../Images/BPAE cells nonsequential_0001.tif', num_channels);
% microscope_sequential = read_tiff('../Images/BPAE cells sequential.tif', num_channels);
% 
% subplot(223)
% imshow(squeeze(microscope_unmixed(:, :, slice, :))*16);
% title('Non Sequential Microscope Unmixed')
% subplot(224)
% imshow(squeeze(microscope_sequential(:, :, slice, :))*16);
% title('Sequential')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
figure
subplot(221)
region_2_non_sequential = read_tiff('BPAE cells nonsequential area2.tif', 3);
imshow(squeeze(region_2_non_sequential(:, :, 7, :))*16);
title('Region 2 Non Sequential Raw')

subplot(222)
Y  = zeros(num_channels, prod(my_scale));%Observation matrix
for ch = 1:num_channels
    for col = 0:num_cols-1
        low = col*my_scale(1)+1;
        high = col*my_scale(1)+my_scale(1);
        Y(ch, low:high) = region_2_non_sequential(:, col+1, 7, ch); 
    end
end
Y0 = Y;
region_2_matrix_unmixed = zeros(512, 512, 3);
H = inv(transpose(Ac) * (Ac+epsilon))*transpose(Ac)*Y0;
for ch = 1:num_channels
    for col = 0:num_cols-1
        low = col*my_scale(1)+1;
        high = col*my_scale(1)+my_scale(1);
        region_2_matrix_unmixed(:, col+1, ch) = H(ch, low:high);%*channel_scales(ch); 
    end
end

imshow(region_2_matrix_unmixed*16);
title('Region 2 Non Sequential Matrix Unmixed')


subplot(223)
[Ac, H] = NNMF(Y, A0, epsilon);
Ac
region_2_NNMF_unmixed = zeros(512, 512, 3);
for ch = 1:num_channels
    for col = 0:num_cols-1
        low = col*my_scale(1)+1;
        high = col*my_scale(1)+my_scale(1);
        region_2_NNMF_unmixed(:, col+1, ch) = H(ch, low:high);%*channel_scales(ch); 
    end
end

imshow(squeeze(region_2_NNMF_unmixed(:, :, :))*16);
title('Region 2 Non Sequential NNMF Unmixed')

subplot(224)
region_2_sequential = read_tiff('BPAE cells sequential area2.tif', 3);
imshow(squeeze(region_2_sequential(:, :, 7, :))*16);
title('Region 2 Sequential')

%}



