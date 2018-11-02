#Adapted from https://github.com/luntai/VGG16_Keras_TensorFlow


from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import plot_model
import cv2
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('tf')
'''
image_dim_ordering in 'th' mode the channels dimension (the depth) is at index 1 (e.g. 3, 256, 256). " \
"In 'tf' mode is it at index 3 (e.g. 256, 256, 3).
'''


def VGG_16():#weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3))) #What to do about this input size
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name = 'content_perception')) #This is the feature representation we want to look at
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    #model.load_weights('/scratch/tmcrae/isonet/perceptual_loss/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights('D:/Projects/Github/Super Resolution/Code/Planar Super Resolution/perceptual_loss/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    #plot_model(model, to_file='VGG-16_model.png')

    layer_name = 'content_perception'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


    return intermediate_layer_model


if __name__ == "__main__":
    im = cv2.resize(cv2.imread('zebra.jpg'),
                    (224, 224)).astype(np.float32)
    # normalization
    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((1, 0, 2))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16()#'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print(out.shape)

    #ordered_index = np.argsort(-out)  # descend sort return index array
    #import CallResult
    #for i in range(0, 3):
    #    print(CallResult.lines[int(ordered_index[0][i])])

