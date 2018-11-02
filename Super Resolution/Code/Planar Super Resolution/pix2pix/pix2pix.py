#https://github.com/eriklindernoren/Keras-GAN#pix2pix
#https://arxiv.org/pdf/1611.07004.pdf


from __future__ import print_function, division
import scipy

from data_loader import DataLoader
from keras import backend as K
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.losses import mean_squared_error
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from perceptual_loss.vgg_16_keras import VGG_16
import datetime
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import time
import numpy as np
import os

def perceptual_loss(y_true, y_pred):
    y_pred = tf.image.central_crop(y_pred, 0.875)
    y_true = tf.image.central_crop(y_true, 0.875)

    #mse_loss = K.mean(mean_squared_error(y_true, y_pred))

    e = VGG_16()
    layers = [l for l in e.layers]
    eval_pred = y_pred
    for i in range(len(layers)):
        eval_pred = layers[i](eval_pred)
    eval_true = y_true
    for i in range(len(layers)):
        eval_true = layers[i](eval_true)
    perceptual_loss = K.mean(mean_squared_error(eval_true, eval_pred))

    #loss = perceptual_loss  + mse_loss

    #perceptual_psnr = - tf.image.psnr(eval_true, eval_pred, K.max(eval_true))

    return perceptual_loss

class Pix2Pix():
    def __init__(self, load_weights = False, timestamp_and_epoch_to_load = None):
        # Input shape
        #self.network_filepath = 'weights/2018_10_29_20_37_55/epoch_96.hdf5'
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.load_weights = load_weights

        # Configure data loader
        self.dataset_name = 'microscopy'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        if self.load_weights:
            network_filepath_discriminator = 'weights/discriminator/'+timestamp_and_epoch_to_load+'.hdf5'
            self.discriminator = load_model(network_filepath_discriminator)
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

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
                              loss_weights=[1, 100],
                              optimizer=optimizer)



    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
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
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        residual = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
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
                                                                        elapsed_time))
                    self.sample_images(epoch, batch_i, is_testing=True)
                    self.sample_images(epoch, batch_i, is_testing=False)
                    filepath = 'weights/generator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                    self.generator.save(filepath)
                    filepath = 'weights/discriminator/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                    self.discriminator.save(filepath)
                    #filepath = 'weights/combined/'+ts+'/epoch_'+str(epoch)+'.hdf5'
                    #self.combined.save(filepath)

    def evaluate(self):
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
    gan = Pix2Pix(load_weights = False)#, timestamp_and_epoch_to_load='2018_10_30_15_43_13/epoch_9')
    #gan.evaluate()
    gan.train(epochs=500, batch_size=8, sample_interval=1)
    print("EOF: Run Successful")