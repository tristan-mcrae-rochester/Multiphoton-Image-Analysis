function unmixed_image = get_fluorophore_from_channel(im_size, fluorophore_indices, signatures, mixed_image, unmixed_image, multi_fluorophore_pixels)

similarities = ones(im_size(4), 1);
for slice = 1:im_size(3)
    for row = 1:im_size(1)
        for col = 1:im_size(2)
            pixel_signature = transpose(squeeze(mixed_image(row, col, slice, :)));
            pixel_intensity = sum(pixel_signature); %Could calculate these some other way
            pixel_signature_normalized = pixel_signature / pixel_intensity;
            for i = 1:length(fluorophore_indices)
                fluorophore = fluorophore_indices(i);
                similarities(fluorophore) = mean((pixel_signature_normalized - signatures(i, :)).^2);
            end
            %similarities = similarities + [0.3; 0];
            sim_diff = similarities(1)-similarities(2);
            if multi_fluorophore_pixels
                inverse_mse = 1./similarities;
                scaled_inverse_mse = inverse_mse/sum(inverse_mse);
                unmixed_image(row, col, slice, :) = pixel_intensity*scaled_inverse_mse;
            else
                [highest_corr, matching_fluorophore] = min(similarities);
                unmixed_image(row, col, slice, matching_fluorophore) = pixel_intensity; 
            end
        end
    end
end

end
