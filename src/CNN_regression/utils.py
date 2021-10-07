import numpy as np
import os
import json
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.statphy.models.percolation import percolation_configuration

def time_to_string(t):
    return t.strftime("%Y.%m.%d.%H.%M.%S")

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    return path

def write_numpy_dic_to_json(dic, path): 
    df = pd.DataFrame(dic) 
    with open(path, 'w') as f:
        df.to_json(f, indent=4)

def print_model_summary(model, path):

    model.summary(print_fn=print)

    with open(make_path(path, 'model_summary.log'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    with open(make_path(path, 'optimizer.json'), 'w') as f:
        json.dump(model.optimizer.get_config(), f, indent=4, sort_keys=True)

def generate_data(dataset_size, lattice_size=128):

    X = []
    y = []

    for _ in range(dataset_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(lattice_size, y[-1]))

    X=np.array(X)
    y=np.array(y)

    X_train = X[:(3*dataset_size)//4, :]
    X_test = X[(3*dataset_size)//4:, :]
    y_train = y[:(3*dataset_size)//4]
    y_test = y[(3*dataset_size)//4:]

    return X_train, X_test, y_train, y_test

def define_callbacks(args):

    callbacks = []
    
    if args.set_lr_scheduler:
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        callbacks += [lr_scheduler]

    if args.set_checkpoint:
        checkpoint_file = make_path(args.stage_train_dir, "ckpt-best.h5")
        checkpoint_cb = ModelCheckpoint(checkpoint_file, 
                                        save_best_only=True, 
                                        monitor='val_loss',
                                        save_weights_only=False) 
        callbacks += [checkpoint_cb]

    if args.set_earlystopping:
        early_stopping_cb = EarlyStopping(patience=20, restore_best_weights=True)
        callbacks += [early_stopping_cb]

    if args.set_tensorboard:
        tensor_board = TensorBoard(log_dir=args.stage_train_dir)
        callbacks += [tensor_board]

    return callbacks