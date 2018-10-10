function write_tiff(image, fname, z_stack, z_first)

%Set defaults for optional parameters
switch nargin
    case 1
        [fname, fpath] =  uiputfile;
        fname = strcat(fpath, fname);
        z_stack = true;
        z_first = false;
    case 2
        z_stack = true;
        z_first = false;
    case 3
        z_first = false;
end

%This function appends images to a tiff so this line deletes anything
%already in your output file to start from scratch
delete(fname)
dims = size(image);

%Populate tiff
if and(length(dims) == 3, ~z_stack)
    imwrite(image, fname)
elseif length(dims) == 3
    if z_first
        for slice = 1:dims(1)
            imwrite(double(squeeze(image(slice, :, :))), fname, 'WriteMode', 'append',  'Compression','none');
        end
    else
        for slice = 1:dims(3)
            imwrite(double(squeeze(image(:, :, slice))), fname, 'WriteMode', 'append',  'Compression','none');
        end
    end
else
    if z_first
        for slice = 1:dims(3)
            for ch = 1:dims(4)
                imwrite(squeeze(image(slice, :, :, ch)), fname, 'WriteMode', 'append',  'Compression','none');
            end
        end
    else
        for slice = 1:dims(3)
            for ch = 1:dims(4)
                imwrite(squeeze(image(:, :, slice, ch)), fname, 'WriteMode', 'append',  'Compression','none');
            end
        end
    end
end


