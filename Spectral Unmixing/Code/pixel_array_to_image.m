function Y = pixel_array_to_image(pixel_array, num_slices, num_rows, num_cols)
dims = size(pixel_array);
num_features = dims(2);


Y  = zeros(num_rows, num_cols, num_slices, num_features);

pixel = 1;
for slice = 1:num_slices
    for row = 1:num_rows
        for col = 1:num_cols
            Y(row, col, slice, :) = pixel_array(pixel, :);
            pixel = pixel+1;
        end
    end
end