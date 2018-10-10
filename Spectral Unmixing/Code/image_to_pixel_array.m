function Y = image_to_pixel_array(image)
dims = size(image);
num_rows = dims(1);
num_cols = dims(2);
num_features = dims(end);
if length(dims) == 3
    num_slices = 1;
else 
    num_slices = dims(3);
end

Y  = zeros(num_rows*num_cols*num_slices, num_features);

%If z-stack
pixel = 1;
if num_slices > 1
    for slice = 1:num_slices
        for row = 1:num_rows
            for col = 1:num_cols
                Y(pixel, :) = image(row, col, slice, :);
                pixel = pixel+1;
            end
        end
    end
else
    %If single image
    for row = 1:num_rows
        for col = 1:num_cols
            Y(pixel, :) = image(row, col, :);
            pixel = pixel+1;
        end
    end
end
end
