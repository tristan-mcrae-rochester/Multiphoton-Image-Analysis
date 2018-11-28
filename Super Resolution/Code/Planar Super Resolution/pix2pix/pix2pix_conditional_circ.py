#https://github.com/eriklindernoren/Keras-GAN#pix2pix
#https://arxiv.org/pdf/1611.07004.pdf


from __future__ import print_function, division
import scipy

#uncomment on CIRC
import matplotlib as mpl
mpl.use('Agg')



from data_loader import DataLoader
from keras import backend as K
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from perceptual_loss.vgg_16_keras import VGG_16
#from pylibtiff.libtiff.tiff import *
import datetime
#import matplotlib.pyplot as plt #Comment out on CIRC
import sys
import tensorflow as tf
import time
import numpy as np
import os
import javabridge as jv
import bioformats as bf
from helper import *

def perceptual_loss(y_true_256, y_pred_256):

    mse_loss = K.mean(mean_squared_error(y_true_256, y_pred_256))
    mae_loss = K.mean(mean_absolute_error(y_true_256, y_pred_256))
    img_nrows, img_ncols = 256, 256

    y_pred = y_pred_256#tf.image.central_crop(y_pred_256, 0.875)
    y_true = y_true_256#tf.image.central_crop(y_true_256, 0.875)


    e = VGG_16()
    layers = [l for l in e.layers]
    eval_pred = y_pred
    for i in range(len(layers)):
        eval_pred = layers[i](eval_pred)
    eval_true = y_true
    for i in range(len(layers)):
        eval_true = layers[i](eval_true)
    perceptual_loss = K.mean(mean_squared_error(eval_true, eval_pred))

    #Total variation loss https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
    a = K.square(y_pred[:, :img_nrows - 1, :img_ncols - 1, :] - y_pred[:, 1:, :img_ncols - 1, :])
    b = K.square(y_pred[:, :img_nrows - 1, :img_ncols - 1, :] - y_pred[:, :img_nrows - 1, 1:, :])
    tv_loss = K.sum(K.pow(a + b, 1.25))


    loss = perceptual_loss  + tf.scalar_mul(0.1, mse_loss) #+ tf.scalar_mul(0.1, tv_loss)

    #perceptual_psnr = - tf.image.psnr(eval_true, eval_pred, K.max(eval_true))

    return loss


class Pix2Pix():
    def __init__(self, load_weights = False, timestamp_and_epoch_to_load = None):
        # Input shape
        #self.network_filepath = 'weights/2018_10_29_20_37_55/epoch_96.hdf5'
        self.img_rows = 256
        self.img_cols = 256
        self.model_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.model_channels)
        self.load_weights = load_weights

        # Configure data loader
        self.dataset_name = 'microscopy_nuclei'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        print("Loading Discriminator", flush = True)
        # Build and compile the discriminator
        if self.load_weights:
            network_filepath_discriminator = 'weights/discriminator/'+timestamp_and_epoch_to_load+'.hdf5'
            print("about to load model", flush = True)
            self.discriminator = load_model(network_filepath_discriminator)
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        print("Loading Generator", flush = True)
        if self.load_weights:
            network_filepath_generator = 'weights/generator/'+timestamp_and_epoch_to_load+'.hdf5'
            self.generator = load_model(network_filepath_generator, custom_objects={'perceptual_loss': perceptual_loss})
        else:
            self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])


        #load weights
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', perceptual_loss],
                              loss_weights=[1, 1],
                              optimizer=optimizer) # loss_weights=[1, 100] 



    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True, dropout_rate = 0):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation=None)(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = Dropout(dropout_rate)(d) #how to apply this when training just the discriminator?
            d = LeakyReLU(alpha=0.2)(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2, dropout_rate=0.5)
        d3 = conv2d(d2, self.gf*4, dropout_rate=0.5)
        d4 = conv2d(d3, self.gf*8, dropout_rate=0.5)
        d5 = conv2d(d4, self.gf*8, dropout_rate=0.5)
        d6 = conv2d(d5, self.gf*8, dropout_rate=0.5)
        d7 = conv2d(d6, self.gf*8, dropout_rate=0.5)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8, dropout_rate=0.5)
        u2 = deconv2d(u1, d5, self.gf*8, dropout_rate=0.5)
        u3 = deconv2d(u2, d4, self.gf*8, dropout_rate=0.5)
        u4 = deconv2d(u3, d3, self.gf*4, dropout_rate=0.5)
        u5 = deconv2d(u4, d2, self.gf*2, dropout_rate=0.5)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        residual = Conv2D(self.model_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        output_img = add([residual, d0]) #K.clip(add([residual, d0]), 0.0, 1.0)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)




    def train(self, epochs, batch_size=1, sample_interval=50):

        ts = time.gmtime()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", ts)
        ts = ts.replace(" ", "_")
        ts = ts.replace(":", "_")
        ts = ts.replace("-", "_")
        os.makedirs('weights/generator/'+ts)
        os.makedirs('weights/discriminator/'+ts)
        os.makedirs('weights/combined/'+ts)
        print("Timestamp: ", flush = True)
        print(ts, flush = True)

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0 and batch_i == 0:
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time), flush = True)
                    #self.sample_images(epoch, batch_i, is_testing=True)
                    #self.sample_images(epoch, batch_i, is_testing=False)

                    if epoch % (sample_interval*100) == 0 and batch_i == 0:
                        #save model
                        filepath = 'weights/generator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                        self.generator.save(filepath)
                        filepath = 'weights/discriminator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                        self.discriminator.save(filepath)

                        #evaluate test data
                        #full_image_save_path = "images/%s/%d_%d_full_train_cell_image.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell #1 area 1 low res.oir', save_path = full_image_save_path, self_ensemble = False, zoom_level = 8, img_channels = 3)
                        #evaluate low res cell validation data
                        #full_image_save_path = "images/%s/%d_%d_full_val_low_res_cell_image.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell area 1 low res.oir', save_path = full_image_save_path, self_ensemble = False, zoom_level = 8, img_channels = 3)
                        #evaluate low low res cell validation data
                        #full_image_save_path = "images/%s/%d_%d_full_val_low_low_res_cell_image.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Low Res/cell area 1 low low res.oir', save_path = full_image_save_path, self_ensemble = False, zoom_level = 8, img_channels = 3)
                        #evaluate Kris and Emma's validation data
                        #full_image_save_path = "images/%s/%d_%d_full_val_Kris_and_Emma_image.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/emma_test_slice_four_channel.tif', save_path = full_image_save_path, self_ensemble = False, zoom_level = 16, img_channels = 4)
                        #full_image_save_path = "images/%s/%d_%d_full_val cell #1 area 9 dapi.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell #1 area 9 low res.tif', save_path = full_image_save_path, self_ensemble = False, zoom_level = 8, img_channels = 1, overlap = False)
                    

                    

    def evaluate(self, full_image_path = None, save_path = "full_output.tif", self_ensemble = False, zoom_level = 1, interpolate_only = False, img_channels = 3, clip_extreme_patches = False, overlap = True):

        if full_image_path != None:
            print(full_image_path)
            buffer_pixels = 0
            filenames_low_res = [full_image_path]
            num_images = len(filenames_low_res)
            #if zoom_level>1:
            size_low_res  = [num_images, int(4096/zoom_level), int(4096/zoom_level), img_channels]
            #else:
            #    size_low_res  = [num_images, 4096, 4096, self.model_channels]
            size_high_res  = [num_images, 4096, 4096, img_channels]
            inputs_low_res = np.empty(size_low_res, np.float64)

            if ".oir" in full_image_path:
                #jv.start_vm(class_path=bf.JARS)

                for i in range(num_images):
                    filename = filenames_low_res[i]
                    rdr = bf.ImageReader(filename, perform_init=True)
                    inputs_low_res[i] = rdr.read(z=0, t=0, series=0, rescale=True)
                
                inputs_low_res = inputs_low_res*16

            if ".tif" in full_image_path:
                inputs_low_res = np.zeros(size_low_res)
                im = Image.open(full_image_path)
                for i in range(img_channels):
                    #import ipdb; ipdb.set_trace()
                    im.seek(i)
                    inputs_low_res[:, :, :, i] = np.array(im)
                #import ipdb; ipdb.set_trace()
                inputs_low_res = inputs_low_res/np.max(inputs_low_res)
                #if num_images == 1:
                #    inputs_low_res = np.expand_dims(inputs_low_res, axis = 0)

                if zoom_level ==1:
                    inputs_high_res = inputs_low_res

            if zoom_level>1:
                inputs_high_res = np.empty(size_high_res, np.float64)
                for i in range(num_images):
                    for c in range(img_channels):
                        #import ipdb; ipdb.set_trace()
                        inputs_high_res[i, :, :, c] = np.clip(zoom(inputs_low_res[i, :, :, c], zoom=zoom_level), 0, 1) #setting upper limit to 0.5 in attempt to eliminate artifacts didn't work

            if overlap:
                tiles_per_image = 961
            else:
                tiles_per_image = 256

            x = np.empty([tiles_per_image*num_images, self.img_rows, self.img_cols, img_channels]) #256
            for c in range(img_channels):
                x[:, :, :, c] = crop(inputs_high_res[:, :, :, c], self.img_rows, self.img_cols, buffer_pixels, overlap=overlap)
                

            if interpolate_only:
                y_pred = x
                

            else:
                y_pred = np.zeros_like(x)
                for c in range(img_channels):
                    single_channel_input = np.empty([tiles_per_image*num_images, self.img_rows, self.img_cols, self.model_channels])
                    for model_channel in range(self.model_channels):
                        single_channel_input[:, :, :, model_channel] = x[:, :, :, c]
                    if self_ensemble:
                        y_pred_single_channel = np.zeros([tiles_per_image*num_images, self.img_rows, self.img_cols])
                        for rotations in range(4):
                            y_pred_update = np.rot90(self.generator.predict(np.rot90(single_channel_input, rotations, axes=(1, 2))), -rotations, axes=(1, 2))
                            y_pred_update = np.median(y_pred_update, axis = -1)
                            if clip_extreme_patches:
                                for max_pixel in [.9, .8, .7, .6, .5, .4, .3, .2, .15]:
                                    peak_intensities = np.max(y_pred_update, axis = (1, 2))
                                    artifact_locations = np.where(peak_intensities >= 1.0)
                                    replacement_predictions = self.generator.predict(np.clip(single_channel_input[artifact_locations], 0, max_pixel))
                                    replacement_predictions = np.median(replacement_predictions, axis = -1)
                                    y_pred_update[artifact_locations] = replacement_predictions


                            y_pred_single_channel = y_pred_single_channel + y_pred_update/8



                        for rotations in range(4):
                            y_pred_update = np.flipud(np.rot90(self.generator.predict(np.rot90(np.flipud(single_channel_input), rotations, axes=(1, 2))), -rotations, axes=(1, 2)))
                            y_pred_update = np.median(y_pred_update, axis = -1)
                            if clip_extreme_patches:
                                for max_pixel in [.9, .8, .7, .6, .5, .4, .3, .2, .15]:
                                    peak_intensities = np.max(y_pred_update, axis = (1, 2))
                                    artifact_locations = np.where(peak_intensities >= 1.0)
                                    replacement_predictions = self.generator.predict(np.clip(single_channel_input[artifact_locations], 0, max_pixel))
                                    replacement_predictions = np.median(replacement_predictions, axis = -1)
                                    y_pred_update[artifact_locations] = replacement_predictions


                            y_pred_single_channel = y_pred_single_channel + y_pred_update/8
                            

                        #y_pred_single_channel = np.mean(y_pred_single_channel, axis = -1)
                    else:
                        y_pred_single_channel = self.generator.predict(single_channel_input)
                        y_pred_single_channel = np.median(y_pred_single_channel, axis = -1)
                        if clip_extreme_patches:
                            for max_pixel in [.9, .8, .7, .6, .5, .4, .3, .2, .15]:
                                peak_intensities = np.max(y_pred_single_channel, axis = (1, 2))
                                artifact_locations = np.where(peak_intensities >= 1.0)
                                #replacement_predictions = self.generator.predict(scipy.signal.medfilt(single_channel_input[artifact_locations], kernel_size = [1, 3, 3, 1]))
                                replacement_predictions = self.generator.predict(np.clip(single_channel_input[artifact_locations], 0, max_pixel))
                                replacement_predictions = np.median(replacement_predictions, axis = -1)
                                y_pred_single_channel[artifact_locations] = replacement_predictions



                    y_pred[:, :, :, c] = y_pred_single_channel



            y_pred_stitched = np.empty([4096, 4096, img_channels])
            for c in range(img_channels):
                y_pred_stitched[:, :, c] = stitch(y_pred[:, :, :, c], [4096, 4096, 1, 1], [self.img_rows, self.img_cols], buffer_pixels, overlap = overlap)

            y_pred_stitched = np.clip(y_pred_stitched, 0, 1)
            y_pred_stitched = (y_pred_stitched*65535).astype('uint16')
            #import ipdb; ipdb.set_trace()
            imlist = []
            for c in range(img_channels):
                #imlist.append(Image.fromarray(inputs_low_res[:, :, c]))
                imlist.append(Image.fromarray(y_pred_stitched[:, :, c]))
            #import ipdb; ipdb.set_trace()

            imlist[0].save(save_path, compression = None, save_all = True, append_images=imlist[1:])
            #im = Image.fromarray(y_pred_stitched)
            #im.save(save_path)

        else:
            self.sample_images(-1, -1, is_testing=True)
            self.sample_images(-1, -1, is_testing=False)


    def sample_images(self, epoch, batch_i, is_testing=True):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=r, is_testing=is_testing)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        gen_imgs = np.clip(gen_imgs, 0, 1)

        titles = ['Low Res', 'High Res Reconstruction', 'High Res Ground Truth']

        plt.figure(1, figsize = (35, 15))
        cnt = 0
        plt.rcParams['font.size'] = 30
        for i in range(c):
            for j in range(r):
                plt.subplot(r, c, cnt+1)
                plt.imshow(gen_imgs[cnt])
                plt.title(titles[i])
                plt.axis('off')
                cnt += 1
        testing = "val" if is_testing else "train"
        plt.savefig("images/%s/%d_%d_%s.png" % (self.dataset_name, epoch, batch_i, testing))
        plt.close()


if __name__ == '__main__':
    jv.start_vm(class_path=bf.JARS)
    print("Initialize gan", flush = True)
    gan = Pix2Pix(load_weights = True, timestamp_and_epoch_to_load='2018_11_20_21_31_13/epoch_1100')


    #2018_11_20_21_31_13/epoch_1100 Overnight training of dapi only dataset for just cell #1. Some noticable tiling artifacts but probably the biggest improvement over interpolation alone I've yet seen.
    #Can make predictions without overlap but tiling is worse. Ensembling helps tiling but makes very blurry. Tiling may get better with time or with more emphasis on mse
    #Seems to have the capacity for enough clarity on its own.
    #Out of dataset performance: Cell area 1. Does decent, but still not as good as in-dataset. Can definitely see adverse effects of interpolating
    #Out of dataset Performance: Mouse Placenta 1. Doesn't die and has some benefits but doesn't match in-dataset performance.


    #Second night of overnight trianing for 5 experiments
    #2018_11_19_18_58_32/epoch_5 bad artifacts
    #2018_11_19_18_58_47/epoch_5
    #2018_11_19_18_59_19/epoch_5 bad artifacts
    #2018_11_19_19_00_25/epoch_5 best contrast of the bunch
    #2018_11_19_19_01_31/epoch_4 bad artifacts

    #First night of overnight training for 5 experiments
    #2018_11_19_18_58_32/epoch_2 Discriminator 1,   Perceptual 10/11, mse 1/11. For Emma's data: it has some minor tiling artifacts compared to 2018_11_12_14_32_11/epoch_40 but lesss extreme artifacts than it. Much less tiling and finer detail 
    #than 2018_11_14_14_19_34/epoch_5 but more background noise. I think difference in background noise is partly just a scaling thing and partly the lack of TV loss. Could potentially benefit from ensembling
    #For Cory's data: Still some extreme artifacts but much less than before 2018_11_12_14_32_11/epoch_40. Significantly more tiling and less sharpness than 2018_11_12_14_32_11/epoch_40. Sharper than 2018_11_14_14_19_34/epoch_5 with about the same tiling
    
    #2018_11_19_18_58_47/epoch_2 Discriminator 1,   Perceptual 10/11, mae 1/11  
    #For Cory's data: still not as sharp as 2018_11_12_14_32_11/epoch_40 but fewer high intensity artifacts than it. Pretty similar overall to 2018_11_19_18_58_32/epoch_2. Very similar to 2018_11_19_18_58_32/epoch_2 for Emma's data as well
    
    #2018_11_19_18_59_19/epoch_2 Discriminator 1,   Perceptual 10/12, mse 2/12 
    #For Cory's data: Maybe a little sharper than 2018_11_14_14_19_34/epoch_5 but not by much. 
    #For Emma's data: Way more tiled than 2018_11_19_18_58_32/epoch_2

    #2018_11_19_19_00_25/epoch_2 Discriminator 0.1, Perceptual 10/11, mse 1/11 
    #For Cory's Data: Very similar to 2018_11_19_18_58_32/epoch_2 without very many extreme artifacts
    #For Emma's Data: Very similar to 2018_11_19_18_58_32/epoch_2

    #2018_11_19_19_01_31/epoch_1 Discriminator 10,  Perceptual 10/11, mse 1/11 
    #For Cory's data: Too Grainey looking. No real improvement in sharpness. No extreme artifacts at all though. And not much tiling issues. Could make a good compliment to 2018_11_12_14_32_11/epoch_40
    #For Emma's data: Very similar to 2018_11_19_18_58_32/epoch_2, maybe a little better background removal but not nearly as good as 2018_11_14_14_19_34/epoch_5


    #2018_11_16_21_14_16/epoch_6 #next 6 epochs of 2018_11_14_14_19_34/epoch_5. More tiling artifacts than before but finally seems to be rid of extreme intensity artifacts. 
    #Blurrier than 2018_11_12_14_32_11/epoch_40, more tiling too. Higher all around intensity than 2018_11_14_14_19_34/epoch_5 without as good contrast. Otherwise quality is about the same
    #I suspect the blurryness and los contrast is coming from TV loss. 

    #2018_11_15_21_18_33/epoch_3 #First 3 epochs of trianing without tv loss on same dataset. Directly comperable to 2018_11_13_21_09_43/epoch_3
    #2018_11_14_14_19_34/epoch_5 #next 5 epochs of same setup #Best on Kris and Emma's data so far but still significant tiling artifacts. Best background removal
    #for Kris and Emma which makes sense because it is trained on noisy images that 2018_11_12_14_32_11/epoch_40 doesn't see
    #Definitely has potential if trained more. Works best on Cory's and Emma's data as of 11/19/18
    #2018_11_13_21_09_43/epoch_3 #First few epochs of expanded training set and single channel prediction. includes dropout and TV loss as well as single channel prediction
    #2018_11_12_14_32_11/epoch_40  #Current best model before I expanded the training set or switched to one channel at a time. 
    #Sharpest outputs by far. Minimal Tiling artifacts. Only issue is high intensity artifacts.

    #2018_11_09_20_17_33/epoch_115
    #2018_11_07_14_47_49/epoch_146
    #gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/cory_original.tiff',              
    #    save_path = "full_output_cory.tif", self_ensemble = False, zoom_level = 4, interpolate_only=False, img_channels = 3, clip_extreme_patches = False) 
    #gan.evaluate(full_image_path = 'D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell area 1 low res.oir',                                   
    #    save_path = "full_output_cell.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 3) 
    #gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/emma_test_slice_four_channel.tif', 
    #    save_path = "full_output_emma.tif", self_ensemble = False, zoom_level = 16, interpolate_only=True, img_channels = 4)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell #1 area 9 low res.tif', 
    #    save_path = "full_output_cell_#1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell area 1 low res.tif', 
    #    save_path = "full_output_cell_area_1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=True, img_channels = 1, overlap = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-mouse placenta area1 low res.tif', 
    #    save_path = "full_output_mouse_placenta_1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=True, img_channels = 1, overlap = False)
    #'D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell area 1 low res.oir' 
    #'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/full_output_57_self_ensemble.tif'
    print("Train", flush = True)
    gan.train(epochs=10000000, batch_size=8, sample_interval=1)
    jv.kill_vm()
    print("EOF: Run Successful")
