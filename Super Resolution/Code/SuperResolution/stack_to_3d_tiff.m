clc; clear all; close all;

load('isonet.mat');


corner_indices = 2:4:2048;
%corner_3d_xz = isonet_test_pred(corner_indices, :, :);
%write_tiff(corner_3d_xz, 'SR_3D_example_XZ_plane.tif')
%corner_3d_xy = permute(corner_3d_xz, [2, 1, 3]);
%write_tiff(corner_3d_xy, 'SR_3D_example_XY_plane.tif')

corner_3d_xz_interp = isonet_test_input(corner_indices, :, :);
%write_tiff(corner_3d_xz_interp, 'SR_3D_example_XZ_plane_interp.tif')

corner_3d_xz_raw = zeros(512, 224, 224);
for i = 1:4:224
    corner_3d_xz_raw(:, i, :) = corner_3d_xz_interp(:, i, :);    
end
write_tiff(corner_3d_xz_raw, 'SR_3D_example_XZ_plane_raw.tif')

%{
full_3d_image = zeros(512, 512, 412); %[y, z, x] Initialize with interpolated inputs instead of zeros
for i = 4:4:2048
   full_3d_image(i/4, 1:224, 1:224) = isonet_test_pred(i-3, :, :);
   full_3d_image(i/4, 1:224, 412-223:412) = isonet_test_pred(i-2, :, :);
   full_3d_image(i/4, 512-223:512, 1:224) = isonet_test_pred(i-1, :, :);
   full_3d_image(i/4, 512-223:512, 412-223:412) = isonet_test_pred(i, :, :);
end
write_tiff(full_3d_image, 'SR_3D_example_full_ZX.tif')
%}