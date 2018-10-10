function signature_matrix = get_fluorophore_signature(fluorophores, channel_ranges)


%fluorophore = 'DAPI';
%fluorophore = 'RFP (TagRFP)';

% ch3 = [420 460];  
% ch2 = [495 540];
% ch1 = [575 630];
% ch4 = [645 685];
% channel_ranges = transpose([ch1; ch2; ch3; ch4]);

num_fluorophores = length(fluorophores);
dims = size(channel_ranges);
num_channels = dims(2);

signature_matrix = zeros(num_fluorophores, num_channels);

for i = 1:num_fluorophores
    fluorophore = fluorophores(i);
    if fluorophore == 'SHG'
        signature = zeros(num_channels, 1);
        signature(i) = 1;
    else
        fname = strcat('D:\Projects\Channel Unmixing\Code\Cornell\', fluorophore, '.csv');
        emission_spectra = csvread(fname, 1, 0);
        signature = zeros(num_channels, 1);
        for ch = 1:num_channels
            ch_min = min(channel_ranges(:, ch));
            ch_max = max(channel_ranges(:, ch));
            X = ch_min:ch_max;
            [unused, emission_min_index] = max(emission_spectra(:, 1) == ch_min);
            [unused, emission_max_index] = max(emission_spectra(:, 1) == ch_max);
            Y = emission_spectra(emission_min_index:emission_max_index, 3);
            if length(Y)<length(X)
               if emission_min_index == 1
                   Y = vertcat(zeros(length(X)-length(Y), 1), Y);
               else
                   Y = vertcat(Y, zeros(length(X)-length(Y), 1));
               end
            end
            intensity = trapz(X, Y);
            signature(ch) = intensity;
        end
        signature = signature/sum(signature);
    end
    signature_matrix(i, :) = signature; 
end
end

