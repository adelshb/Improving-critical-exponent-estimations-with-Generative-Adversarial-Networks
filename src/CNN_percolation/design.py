# -*- coding: utf-8 -*-
#
# Written by Hor (Ebi) Dashti, https://github.com/h-dashti
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from tensorflow import keras

def create_model(input_shape, K, 
                 n_conv_layers=4, 
                 n_dense_layers=3, 
                 n_neurons=512, 
                 dropout_rate=0):

    # input layer
    i = keras.layers.Input(shape=input_shape)
    x = i

    # Convolution block
    for l in range(n_conv_layers):
        n_filters = 32 * (2 ** l)
        x = keras.layers.Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D((2,2))(x)


    x = keras.layers.Flatten()(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)

    # Classification block
    for l in range(n_dense_layers):
        x = keras.layers.Dense(n_neurons, activation='relu')(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)
            
    x = keras.layers.Dense(K, activation='softmax')(x)
    model = keras.models.Model(i, x)
    return model
    