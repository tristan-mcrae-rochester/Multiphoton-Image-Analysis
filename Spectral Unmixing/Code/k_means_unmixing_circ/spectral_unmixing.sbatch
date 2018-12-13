#!/bin/sh
#SBATCH -p standard
#SBATCH --mem=50GB
#SBATCH -t 5-0:00:00   
#SBATCH -J spectral_unmixing_job
#SBATCH -o spectral_unmixing.out

module load matlab

matlab -nodesktop -nodisplay -nosplash -singleCompThread -r "
num_channels = 3; num_timesteps = 1; num_fluorophores_to_unmix = 3; channels_to_unmix = [1, 2, 3]; load_file = '112018 D7 A Ca lung NT cd31.oif'; save_method = 'single_tiff'; save_file = 'testing_sbatch_large'; KMeansUnmixing(num_channels, num_timesteps, num_fluorophores_to_unmix, channels_to_unmix, load_file, save_method, save_file); exit"


