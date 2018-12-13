#https://github.com/eriklindernoren/Keras-GAN#pix2pix
#https://arxiv.org/pdf/1611.07004.pdf


from __future__ import print_function, division
import scipy

#uncomment on CIRC
#import matplotlib as mpl
#mpl.use('Agg')



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
import matplotlib.pyplot as plt #Comment out on CIRC
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
    def __init__(self, load_weights = False, timestamp_and_epoch_to_load = None, interpolated = False):
        # Input shape
        self.interpolated = interpolated
        self.img_rows = 256
        self.img_cols = 256
        if not self.interpolated:
            self.img_rows = 64
            self.img_cols = 64
        self.model_channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.model_channels)
        self.load_weights = load_weights
        self.num_gpus = 1
        

        # Configure data loader
        self.dataset_name = 'microscopy_med_res_nuclei_uninterpolated'
        #self.dataset_name = 'microscopy_med_res_nuclei_interpolated'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (16, 16, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if self.load_weights:
            network_filepath_discriminator = 'weights/discriminator/'+timestamp_and_epoch_to_load+'.hdf5'
            self.discriminator = load_model(network_filepath_discriminator)
            if self.num_gpus>1:
                self.discriminator_gpu = multi_gpu_model(self.discriminator, gpus = self.num_gpus)
        else:
            self.discriminator = self.build_discriminator()
            if self.num_gpus>1:
                self.discriminator_gpu = multi_gpu_model(self.discriminator, gpus = self.num_gpus)
            self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        #print(self.discriminator.summary())

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        if self.load_weights:
            network_filepath_generator = 'weights/generator/'+timestamp_and_epoch_to_load+'.hdf5'
            self.generator = load_model(network_filepath_generator, custom_objects={'perceptual_loss': perceptual_loss})
        else:
            self.generator = self.build_generator()
        if self.num_gpus>1:
            self.generator_gpu = multi_gpu_model(self.generator, gpus = self.num_gpus)
        #Sprint(self.generator.summary())
        # Input images and their conditioning images
        img_A = Input(shape=(256, 256, 3))
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        if self.num_gpus>1:
            fake_A = self.generator_gpu(img_B)
            # For the combined model we will only train the generator
            self.discriminator_gpu.trainable = False
            # Discriminators determines validity of translated images / condition pairs
            valid = self.discriminator_gpu([fake_A, img_B])
            self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
            self.combined_gpu = multi_gpu_model(self.combined, gpus = self.num_gpus)
            self.combined_gpu.compile(loss=['mse', perceptual_loss],
                              loss_weights=[1, 1],
                             optimizer=optimizer) 
        else:
            fake_A = self.generator(img_B)
            # For the combined model we will only train the generator
            self.discriminator.trainable = False
            # Discriminators determines validity of translated images / condition pairs
            valid = self.discriminator([fake_A, img_B])
            self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
            self.combined.compile(loss=['mse', perceptual_loss],
                              loss_weights=[1, 1],
                             optimizer=optimizer) 



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

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.5):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        def neural_upsampling(layer_input, filters, f_size=4, dropout_rate=0.5, name = None):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu', name = name)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        if self.interpolated:
            # Downsampling
            d1 = conv2d(d0, self.gf, bn=False, dropout_rate=0.0)
            d2 = conv2d(d1, self.gf*2)
            d3 = conv2d(d2, self.gf*4)
            d4 = conv2d(d3, self.gf*8)
            d5 = conv2d(d4, self.gf*8)
            d6 = conv2d(d5, self.gf*8)
            d7 = conv2d(d6, self.gf*8)

            # Upsampling
            u1 = deconv2d(d7, d6, self.gf*8)
            u2 = deconv2d(u1, d5, self.gf*8)
            u3 = deconv2d(u2, d4, self.gf*8)
            u4 = deconv2d(u3, d3, self.gf*4)
            u5 = deconv2d(u4, d2, self.gf*2)
            u6 = deconv2d(u5, d1, self.gf, dropout_rate=0.0)

            u7 = UpSampling2D(size=2)(u6)
            residual = Conv2D(self.model_channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
            output_img = add([residual, d0]) #K.clip(add([residual, d0]), 0.0, 1.0)

        else:
            # Downsampling
            d1 = conv2d(d0, self.gf, bn=False, dropout_rate=0.2) #32
            d2 = conv2d(d1, self.gf*2) #16
            d3 = conv2d(d2, self.gf*4) #8
            d4 = conv2d(d3, self.gf*8) #4
            d5 = conv2d(d4, self.gf*8) #2
            d6 = conv2d(d5, self.gf*8) #1

            # Upsampling
            u1 = deconv2d(d6, d5, self.gf*8)                        #2
            u2 = deconv2d(u1, d4, self.gf*8)                        #4
            u3 = deconv2d(u2, d3, self.gf*8)                        #8
            u4 = deconv2d(u3, d2, self.gf*4)                        #16
            u5 = deconv2d(u4, d1, self.gf*2)                        #32
            u6 = neural_upsampling(u5, self.gf, name = "u6")        #64
            u7 = neural_upsampling(u6, self.gf, name = "u7")        #128
            u8 = neural_upsampling(u7, self.gf, name = "u8")        #256
            

            output_img = Conv2D(self.model_channels, kernel_size=4, strides=1, padding='same', activation='tanh', name = "output_image")(u8)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d


        #Is this getting the correct inputs?
        img_A = Input(shape=(256, 256, 3), name = 'img_A')
        img_B = Input(shape=self.img_shape, name = 'img_B')
        # Concatenate image and conditioning image by channels to produce input
        if self.interpolated:
            combined_imgs = Concatenate(axis=-1)([img_A, img_B])
        else:
            upsampled_img_B = UpSampling2D(size=4)(img_B)
            combined_imgs = Concatenate(axis=-1)([img_A, upsampled_img_B])
    
        
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)





        return Model([img_A, img_B], validity)




    def train(self, epochs, batch_size=1, sample_interval=50, fuzzy_labels = False, noisy_inputs = False):

        ts = time.gmtime()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", ts)
        ts = ts.replace(" ", "_")
        ts = ts.replace(":", "_")
        ts = ts.replace("-", "_")
        os.makedirs('weights/generator/'+ts)
        os.makedirs('weights/discriminator/'+ts)
        os.makedirs('weights/combined/'+ts)

        print(ts, flush = True)


        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid_ones = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        if fuzzy_labels:
            valid = valid_ones * 0.9
        else:
            valid = valid_ones

        for epoch in range(epochs):
            batches = list(self.data_loader.load_batch(batch_size, interpolation = self.interpolated))[0]
            batches_A = np.asarray(batches[0])
            batches_B = np.asarray(batches[1])
            
            for batch_i in range(len(batches_A)):
                imgs_A = batches_A[batch_i]
                imgs_B = batches_B[batch_i]
                

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                if self.num_gpus>1:
                    fake_A = self.generator_gpu.predict(imgs_B)
                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator_gpu.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator_gpu.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    if noisy_inputs:
                        input_noise = np.random.normal(loc = 0.0, scale = 0.05, size = imgs_B.shape)
                        g_loss = self.combined_gpu.train_on_batch([imgs_A, imgs_B+input_noise], [valid, imgs_A])
                    else:
                        g_loss = self.combined_gpu.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                else:
                    fake_A = self.generator.predict(imgs_B)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    if noisy_inputs:
                        input_noise = np.random.normal(loc = 0.0, scale = 0.001, size = imgs_B.shape)
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B+input_noise], [valid, imgs_A])
                    else:
                        g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])


                ###coppied from unconditional GAN start
                
                # If at save interval => save generated image samples
                if epoch % sample_interval == 0 and batch_i == 0:
                    d_loss_real = 0
                    d_acc_real  = 0
                    d_loss_fake = 0
                    d_acc_fake  = 0
                    d_penalty   = 0
                    g_loss      = 0
                    c_loss      = 0


                    batches_eval = list(self.data_loader.load_batch(batch_size, interpolation = self.interpolated))[0]
                    batches_A_eval = np.asarray(batches[0])
                    batches_B_eval = np.asarray(batches[1])
                    num_batches = len(batches_A)
                    for batch_i_eval in range(num_batches):
                        imgs_A_eval = batches_A_eval[batch_i_eval]
                        imgs_B_eval = batches_B_eval[batch_i_eval]
                        fake_A_eval = self.generator.predict(imgs_B_eval)

                        
                        d_stats_real = self.discriminator.evaluate([imgs_A_eval, imgs_B_eval], valid_ones, verbose = 0)

                        d_loss_real = d_loss_real + d_stats_real[0]/num_batches
                        d_acc_real = d_acc_real + d_stats_real[1]/num_batches

                        d_stats_fake = self.discriminator.evaluate([fake_A_eval, imgs_B_eval], fake, verbose = 0)
                        d_loss_fake = d_loss_fake + d_stats_fake[0]/num_batches
                        d_acc_fake = d_acc_fake + d_stats_fake[1]/num_batches
                        


                        
                        c_stats = self.combined.evaluate([imgs_A_eval, imgs_B_eval], [valid_ones, imgs_A], verbose = 0)
                        d_penalty = d_penalty + c_stats[1]/num_batches
                        g_loss = g_loss + c_stats[2]/num_batches
                        c_loss = c_loss + c_stats[0]/num_batches


                    elapsed_time = datetime.datetime.now() - start_time
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss real: %f, acc real: %3d%%] [D loss fake: %f, acc fake: %3d%%] [D Penalty: %f] [G Loss: %f] [C loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss_real, 100*d_acc_real, d_loss_fake, 100*d_acc_fake, d_penalty,
                                                                        g_loss, c_loss,
                                                                        elapsed_time), flush = True)               


                    ###coppied from unconditional GAN end
                    self.sample_images(epoch, batch_i, is_testing=False)
                    '''
                    elapsed_time = datetime.datetime.now() - start_time
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                    #self.sample_images(epoch, batch_i, is_testing=True)
                    #self.sample_images(epoch, batch_i, is_testing=False)
                    '''
                    if epoch % (sample_interval*100) == 0 and batch_i == 0:
                        #save model
                        filepath = 'weights/generator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                        self.generator.save(filepath)
                        filepath = 'weights/discriminator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                        self.discriminator.save(filepath)
                        #full_image_save_path = "images/%s/%d_%d_full_val cell #1 area 9 fibers.tif" % (self.dataset_name, epoch, batch_i)
                        #self.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C2-cell #1 area 9 low res.tif', save_path = full_image_save_path, self_ensemble = False, zoom_level = 8, img_channels = 1, overlap = False)

                    

                    

    def evaluate(self, full_image_path = None, save_path = "full_output.tif", self_ensemble = False, zoom_level = 1, interpolate_only = False, img_channels = 3, clip_extreme_patches = False, overlap = True, img_dimension = 4096):

        if full_image_path != None:
            print(full_image_path, flush = True)
            buffer_pixels = 0
            filenames_low_res = [full_image_path]
            num_images = len(filenames_low_res)
            #if zoom_level>1:
            size_low_res  = [num_images, int(img_dimension/zoom_level), int(img_dimension/zoom_level), img_channels]
            #else:
            #    size_low_res  = [num_images, 4096, 4096, self.model_channels]
            size_high_res  = [num_images, img_dimension, img_dimension, img_channels]
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
                #scale intensities before putting into network
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


            if False:
                single_channel_input = np.zeros([1, img_dimension, img_dimension, 3])
                for model_channel in range(self.model_channels):
                    single_channel_input[0, :, :, model_channel] = inputs_high_res[0, :, :, 0]
                y_pred = self.generator.predict(single_channel_input)
                y_pred = np.clip(y_pred, 0, 1)
                y_pred = (y_pred*65535).astype('uint16')
                imlist = []
                imlist.append(Image.fromarray(y_pred[:, :, 1]))

                imlist[0].save(save_path, compression = None, save_all = True, append_images=imlist[1:])

                return


            if overlap:
                if self.interpolated:
                    tiles_per_image = 961
                else:
                    tiles_per_image = 225
            else:
                if self.interpolated:
                    tiles_per_image = 256
                else:
                    tiles_per_image = 64


            x = np.empty([tiles_per_image*num_images, self.img_rows, self.img_cols, img_channels]) #256
            for c in range(img_channels):
                x[:, :, :, c] = crop(inputs_high_res[:, :, :, c], self.img_rows, self.img_cols, buffer_pixels, overlap=overlap)
                

            if interpolate_only:
                y_pred = x
                

            else:
                y_pred = np.zeros([tiles_per_image*num_images, 256, 256, img_channels])
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


            
            y_pred_stitched = np.empty([2048, 2048, img_channels])
            for c in range(img_channels):
                y_pred_stitched[:, :, c] = stitch(y_pred[:, :, :, c], [2048, 2048, 1, 1], [256, 256], buffer_pixels, overlap = overlap)


            import ipdb; ipdb.set_trace()
            y_pred_stitched = np.clip(y_pred_stitched, 0, 1)
            y_pred_stitched = (y_pred_stitched*65535).astype('uint16')
            imlist = []

            for c in range(img_channels):
                imlist.append(Image.fromarray(y_pred_stitched[:, :, c]))

            imlist[0].save(save_path, compression = None, save_all = True, append_images=imlist[1:])

        else:
            self.sample_images(-1, -1, is_testing=True)
            self.sample_images(-1, -1, is_testing=False)


    def sample_images(self, epoch, batch_i, is_testing=True):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=r, is_testing=is_testing, interpolation = self.interpolated)
        #import ipdb; ipdb.set_trace()
        test = np.concatenate(imgs_B, axis = 0)
        fake_A = self.generator.predict(np.expand_dims(test, axis = 0)) #np.expand_dims(imgs_B, axis = 0)


        #import ipdb; ipdb.set_trace()
        #imgs_A[0] = scipy.misc.imresize(imgs_A[0], (256, 256))
        imgs_B[0] = scipy.misc.imresize(imgs_B[0], (256, 256))/255.0
        #fake_A[0] = scipy.misc.imresize(fake_A[0], (256, 256))

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        gen_imgs = np.clip(gen_imgs, 0, 1)

        #import ipdb; ipdb.set_trace()

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
    gan = Pix2Pix(load_weights = False, interpolated = False)#, timestamp_and_epoch_to_load='2018_12_04_21_42_20/epoch_9000')
    

    #2018_12_04_21_42_20/epoch_9000    six hours of non-interpolated medium res nuclei trianing




    #fiber_gan = Pix2Pix(load_weights = True, timestamp_and_epoch_to_load='2018_12_01_17_23_36/epoch_1130')
    #nuclei_gan = Pix2Pix(load_weights = True, timestamp_and_epoch_to_load='2018_12_01_17_24_10/epoch_250')

    #Training from Saturday morning till Monday morning
    #2018_12_01_17_33_42/epoch_8700    Tesla V100  #default nuclei the entire thing is high intensity artifacts basically
    #2018_12_01_20_28_29/epoch_2000    Tesla K80   #default fibers lots of high intensity artifacts
    #2018_12_01_17_26_06/epoch_400     Tesla K20Xm #random noise nuclei again, tons of high intensity artifacts otherwise better than default
    #2018_12_01_17_26_03/epoch_2000    Tesla K20Xm #random noise fibers again, tons of high intensity artifacts otherwise better than default
    #2018_12_01_17_27_20/epoch_400     Tesla K20Xm #fuzzy labels nuclei awful at empty space again. Areas with something better than default
    #2018_12_01_17_30_03/epoch_1500    Tesla K20Xm #fuzzy labels fibers awful at empty space again.

    #earlier epochs for apples to apples comparison
    #2018_12_01_17_33_42/epoch_400     Tesla V100  #default nuclei #looks absolutely terrible
    #2018_12_01_20_28_29/epoch_1500    Tesla K80   #default fibers #lots of high intensity artifacts but sort of smooth. Not much tiling artifacts
    
    #2018_11_28_19_15_27/epoch_700    #loaded from 2018_11_12_14_32_11/epoch_40 trained on ch1 only. Looks pretty good. Definitely better than interpolated. One minor high intensity artifact.
    #2018_11_28_14_21_42/epoch_800    #fibers. Clearer, straighter lines with fewer stitching and high intensity artifacts.
    #2018_11_27_14_06_03/epoch_1700   #dapi. Looks very similar to epoch 800 of same network. 


    #2018_11_28_14_21_42/epoch_200 fewer extreme artifacts than epoch 100. a little fewer stitching artifacts too. Maybe slightly more small scale pattern artifacts
    #2018_11_28_14_21_42/epoch_100 loading from 2018_11_12_14_32_11/epoch_40 and training on fibers only. High intensity artifacts are prevelent. Fibers are pretty sharp but a bit wavy. Sharper than 2018_11_12_14_32_11/epoch_40. Keep training

    #2018_11_27_13_56_01/epoch_900 attempt at re-creating 1/20 results with model starting from scratch. Trained overnight. Looks decent. Similar to 2018_11_20_21_31_13/epoch_1100. kill
    #2018_11_27_14_06_03/epoch_800 attempt at re-creating 1/20 results with model starting from 2018_11_12_14_32_11/epoch_40. Trained overnight. Similar to 2018_11_27_14_06_03/epoch_200 but with fewer small scale artifacts
    #2018_11_27_19_19_04/epoch_700 attempt at re-creating 1/20 results with model starting from 2018_11_20_21_31_13/epoch_1100. Trained overnight. Almost identical to 2018_11_20_21_31_13/epoch_1100. kill


    #2018_11_27_14_06_03/epoch_200  #attempt at re-creating 1/20 results with model starting from 2018_11_12_14_32_11/epoch_40. Some small scale local artifacts but overall not too bad, may actually benefit from TV loss. Higher contrast than 11/20 version. Probably needs more training time.


    #Runs trained over Thanksgiving
    #2018_11_21_18_51_22/epoch_4500
    #2018_11_21_18_56_48/epoch_3900
    #2018_11_21_18_56_48/epoch_1100
    #2018_11_21_18_57_14/epoch_4500
    #2018_11_21_19_40_03/epoch_4800
    #2018_11_21_19_23_23/epoch_4400
    #2018_11_21_19_21_09/epoch_4400


    #2018_11_20_21_31_13/epoch_1100 Overnight training of dapi only dataset for just cell #1. Some noticable tilign artifacts but probably the biggest improvement over interpolation alone I've yet seen.
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
    #fiber_gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/C1-cory_original.tif',              
    #    save_path = "C1_output_cory.tif", self_ensemble = False, zoom_level = 4, interpolate_only=True, img_channels = 1, clip_extreme_patches = False) 
    #fiber_gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/C2-cory_original.tif',              
    #    save_path = "C2_output_cory.tif", self_ensemble = False, zoom_level = 4, interpolate_only=True, img_channels = 1, clip_extreme_patches = False) 
    #nuclei_gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/C3-cory_original.tif',              
    #    save_path = "C3_output_cory.tif", self_ensemble = False, zoom_level = 4, interpolate_only=True, img_channels = 1, clip_extreme_patches = False) 
    #gan.evaluate(full_image_path = 'D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell area 1 low res.oir',                                   
    #    save_path = "full_output_cell.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 3) 
    #gan.evaluate(full_image_path = 'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/Test outputs/emma_test_slice_four_channel.tif', 
    #    save_path = "full_output_emma.tif", self_ensemble = False, zoom_level = 16, interpolate_only=True, img_channels = 4)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell #1 area 8 low res.tif', 
    #    save_path = "full_output_cell_nuclei_train_#1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = False)
    #nuclei_gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell #1 area 9 low res.tif', 
    
    #    save_path = "full_output_cell_nuclei_#1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = True)
    #fiber_gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C2-cell #1 area 9 low res.tif', 
    #    save_path = "full_output_cell_fibers_#1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = True, clip_extreme_patches = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C1-cell #1 area 9 low res.tif', 
    #    save_path = "full_output_cell_ch1_#1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = True, clip_extreme_patches = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-cell area 1 low res.tif', 
    #    save_path = "full_output_cell_area_1_ch3.tif", self_ensemble = False, zoom_level = 8, interpolate_only=False, img_channels = 1, overlap = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C2-cell area 1 low res.tif', 
    #    save_path = "full_output_cell_area_1_ch2.tif", self_ensemble = False, zoom_level = 8, interpolate_only=True, img_channels = 1, overlap = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C1-cell area 1 low res.tif', 
    #    save_path = "full_output_cell_area_1_ch1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=True, img_channels = 1, overlap = False)
    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/C3-mouse placenta area1 low res.tif', 
    #    save_path = "full_output_mouse_placenta_1.tif", self_ensemble = False, zoom_level = 8, interpolate_only=True, img_channels = 1, overlap = False)

    #gan.evaluate('D:/Projects/Local/Super Resolution/Planar SR Images/2048 nuclei training images/human lung 2 area3 L.tif', 
    #    save_path = "medium_res_nuclei_human_lung_2_area_4.tif", self_ensemble = False, zoom_level = 1, interpolate_only=False, img_channels = 1, overlap = False, img_dimension = 512)


    #'D:/Projects/Local/Super Resolution/Planar SR Images/Low Res/cell area 1 low res.oir' 
    #'D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/pix2pix/full_output_57_self_ensemble.tif'
    gan.train(epochs=10000, batch_size=8, sample_interval=1, fuzzy_labels = False, noisy_inputs = False)


    jv.kill_vm()
    print("EOF: Run Successful", flush = True)