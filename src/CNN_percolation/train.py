
import os, sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import design
import utils

def train(X, y,
          random_state=42,
          test_size=0.20,
          stage_train_dir='.',
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

      
    N, L = X.shape[0], X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, 
                                                        ) #stratify=y
    K = len(np.unique(y_train))  
    

    # design the architecture of model
    model = design.create_model((L, L, 1), K, dropout_rate=dropout_rate)

    # compile the model            
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['sparse_categorical_accuracy'])
        

    # Callbacks  
    callbacks = []
    
    if set_lr_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        callbacks += [lr_scheduler]

    if set_checkpoint:
        checkpoint_file = utils.make_path(stage_train_dir, "ckpt-best.h5")
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


    # print model info
    if dump_model_summary:
        utils.print_model_summary(model, stage_train_dir)
        

    # training the model
    history = model.fit(X_train, y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=epochs,
                        batch_size=batch_size)

    if save_model:
        model.save(utils.make_path(stage_train_dir, 'saved-model.h5'))


    if dump_history:
        utils.write_numpy_dic_to_json(history.history, 
                                    utils.make_path(stage_train_dir, 'history.json')
                                    )
    
    
    loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
    print('loss_test={:.3f}, accuracy_test={:.3f}'.format(loss_test, accuracy_test))

    """loaded_model = tf.keras.models.load_model(make_path(stage_train_dir, 'saved-model.h5'))
    loss_test, accuracy_test = loaded_model.evaluate(X_test, y_test, verbose=0)
    print('loaded_model_loss_test={:.3f}, loaded_model_accuracy_test={:.3f}'.format(loss_test, accuracy_test))"""
    
    
    return model, history
   

#====================================================================
