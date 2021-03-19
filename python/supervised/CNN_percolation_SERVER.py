import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from percolation import read_percolation_data
from sklearn.model_selection import train_test_split
import sys

L = 128
pc = 0.59274
p_arr = np.round(np.arange(0, 0.01, 0.01), 4)

X, y = read_percolation_data(L, p_arr, pc, max_configs_per_p=1000)

N = X.shape[0]
L = X.shape[1] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

K = len(set(y_train))

def CNN_net(L, K):
    # input layer
    i = Input(shape=(L,L,1))

    # Convolution block
    x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # Classification block
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(K, activation='softmax')(x)

    model = Model(i, x)
    return model

model_phase = CNN_net(L, K)

# Inverse time decaying learning rate
# initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
initial_learning_rate = 0.01
decay_steps = 1.0
decay_rate = 0.5
learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
#opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model_phase.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')

r_phase = model_phase.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2)

# We save the model
model_phase.save('./test.h5')