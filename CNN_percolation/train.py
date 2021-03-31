# %%
import os, sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import design
import utils
#====================================================================
#====================================================================

def train(X, y,
          random_state=42,
          test_size=0.20,
          stage_train_dir='.',
          n_gpus=1,
          patience=10,
          epochs=10,
          check_checkpoint=True,
          check_earlystopping=True,
          check_tensorboard=False,
          **kwargs
         ):

      
    # %%
    N, L = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, 
                                                        ) #stratify=y
    K = len(np.unique(y_train))  

    
    # GPU distribution
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)
    devices_names = [d.name.split('e:')[1] for d in devices]
    strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])

    with strategy.scope():
        # %%
        model = design.create_model(L, K)


        # %%
        # Compiling the model
        initial_learning_rate = 0.01
        decay_steps = 1.0
        decay_rate = 0.5
        learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate, decay_steps, decay_rate)

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # %%
    # Callbacks
    callbacks = []

    if check_checkpoint:
        checkpoint_file = os.path.join(stage_train_dir, "ckpt-best.h5")
        checkpoint_cb = ModelCheckpoint(checkpoint_file, save_best_only=True) #
        callbacks += [checkpoint_cb]
    
    if check_earlystopping:
        early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights=True)
        callbacks += [early_stopping_cb]

    if check_tensorboard:
        tensor_board = TensorBoard(log_dir=stage_train_dir)
        callbacks += [tensor_board]


    # %%
    # training the model
    history = model.fit(X_train, y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=epochs)

    if not check_checkpoint:
        model_path = os.path.join(stage_train_dir, 'saved-model.h5')
        model.save(model_path)

    return model, history
   

#====================================================================
