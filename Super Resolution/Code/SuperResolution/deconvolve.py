#http://scikit-image.org/docs/dev/auto_examples/filters/plot_restoration.html

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from scipy.signal import fftconvolve as scipycnv
from scipy.ndimage import zoom


def convolution_op(ground , psf):
    sz = ground.shape[2]
    convolved = []
    for i in range(ground.shape[2]):
        convolved.append(np.real(scipycnv(ground[:,:,i], psf, mode='same')))
    return np.stack(convolved,axis=-1)

def generate_training_samples(ground, psf, ratio):
    sx, sy, sz = ground.shape
    p_xy = convolution_op(ground , psf)
    p_xy = zoom(p_xy, (1/ratio, 1, 1))
    p_xy = zoom(p_xy, (sx/p_xy.shape[0], 1, 1))
    d = {'x' : [], 'y' : []}
    for i in range(sz):
        d['y'].append(ground[:,:,i])
        d['x'].append(p_xy[:,:,i])
    d['x'] = np.stack(d['x'],axis=-1)
    d['y'] = np.stack(d['y'],axis=-1)
    d['x'] = d['x'].transpose(2, 0, 1)
    d['y'] = d['y'].transpose(2, 0, 1)
    return d

def read_from_mat(file):
    mat = spio.loadmat(file , squeeze_me=True)
    return mat


microns_per_pixel_psf = 1.0#1.0/30.0     
microns_per_pixel_image = 1.0
microns_per_voxel_image = 2.0 

axial_to_lateral_ratio = microns_per_voxel_image/microns_per_pixel_image 

#Load in data
ground = read_from_mat('ground.mat')['ground'].astype(np.float32)
psf =  read_from_mat('psf.mat')['psf']
psf = psf[:, :-1, 22]
ground = ground[:, :-1, :] #remove this line
psf = zoom(psf, microns_per_pixel_psf/microns_per_pixel_image)

#Generate training and testing samples
d = generate_training_samples(ground, psf, axial_to_lateral_ratio)
x_train_full, y_train_full = d['x'], d['y']

blurred_dict = {'blurred_images': x_train_full, 'psf': psf, 'ground_images': ground}

spio.savemat("deconv_images/blurred_images.mat", blurred_dict)


'''
astro = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()
'''