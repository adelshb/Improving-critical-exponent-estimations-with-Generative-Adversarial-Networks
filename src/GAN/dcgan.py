# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(2*2*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((2, 2, 256)))
    assert model.output_shape == (None, 2, 2, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        
    def sign_fct(x):
        import tensorflow as tf
        return tf.sign(x)
        
    model.add(layers.Lambda(lambda x: sign_fct(x)))
    assert model.output_shape == (None, 128, 128, 1)

    return model

def make_cnn_model(input_shape=(128, 128, 1), 
                   K=41, 
                   n_conv_layers=4, 
                   n_dense_layers=3, 
                   n_neurons=512, 
                   dropout_rate=0):

    # input layer
    i = tf.keras.layers.Input(shape=input_shape)
    x = i

    # Convolution block
    for l in range(n_conv_layers):
        n_filters = 32 * (2 ** l)
        x = tf.keras.layers.Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)


    x = tf.keras.layers.Flatten()(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Classification block
    for l in range(n_dense_layers):
        x = tf.keras.layers.Dense(n_neurons, activation='relu')(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            
    x = tf.keras.layers.Dense(K, activation='softmax')(x)
    model = tf.keras.models.Model(i, x)
    return model

