
import os, sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import design
import utils
from utils import make_path
import json
#====================================================================
def print_model_summary(model, stage_train_dir):

    model.summary(print_fn=print)

    with open(make_path(stage_train_dir, 'model_summary.log'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        


    # print optimizations
    with open(make_path(stage_train_dir, 'optimizer.json'), 'w') as f:
        json.dump(model.optimizer.get_config(), f, indent=4, sort_keys=True)
#====================================================================


def create_and_compile_model(input_shape, K, dropout_rate):

    model = design.create_model(input_shape, K, dropout_rate=dropout_rate)

    # Compiling the model            
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    return model

#====================================================================

def train(X, y,
          random_state=42,
          test_size=0.20,
          stage_train_dir='.',
          n_gpus=1,
          patience=10,
          epochs=10,
          batch_size=None,
          dropout_rate=0.2,

          dump_model_summary=True,
          set_lr_scheduler=True,
          set_checkpoint=True,
          set_earlystopping=True,
          set_tensorboard=False,
          dump_history=True,
          save_model=True,
          **kwargs
         ):

      
    # %%
    N, L = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, 
                                                        ) #stratify=y
    K = len(np.unique(y_train))  
    

    
    # %%
    # GPU distribution
    device_type = 'GPU'
    devices = tf.config.experimental.list_physical_devices(device_type)

    if len(devices) > 1 and n_gpus > 1 :
        #devices_names = [d.name.split('e:')[1] for d in devices]
        #strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = create_and_compile_model((L,L,1), K, dropout_rate)
    else:
        model = create_and_compile_model((L,L,1), K, dropout_rate)

  

    # %%
    # Callbacks
    
    callbacks = []
    
    if set_lr_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        callbacks += [lr_scheduler]

    if set_checkpoint:
        checkpoint_file = make_path(stage_train_dir, "ckpt-best.h5")
        checkpoint_cb = ModelCheckpoint(checkpoint_file, 
                                        save_best_only=True, 
                                        monitor='val_loss',
                                        save_weights_only=False) 
        callbacks += [checkpoint_cb]
    
    if set_earlystopping:
        early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights=True)
        callbacks += [early_stopping_cb]

    if set_tensorboard:
        tensor_board = TensorBoard(log_dir=stage_train_dir)
        callbacks += [tensor_board]


    # %%
    # print model info
    if dump_model_summary:
        print_model_summary(model, stage_train_dir)
        
        

    # %%
    # training the model
    history = model.fit(X_train, y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=epochs,
                        batch_size=batch_size)

    if save_model:
        model.save(make_path(stage_train_dir, 'saved-model.h5'))


    if dump_history:
        utils.write_numpy_dic_to_json(history.history, 
                                    make_path(stage_train_dir, 'history.json')
                                    )
       

    return model, history
   

#====================================================================
