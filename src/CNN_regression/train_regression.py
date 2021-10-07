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

from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.statphy.models.percolation import percolation_configuration
from src.CNN_regression.design_regression import create_model
from src.CNN_regression.utils import *

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

def main(args):

    start_time = datetime.now()

    X_train, X_test, y_train, y_test = generate_data(args.dataset_size, args.lattice_size)
    
    stage_train_dir = make_path(args.odir, time_to_string(start_time)) 
    os.makedirs(stage_train_dir, exist_ok=True)

    with open(make_path(stage_train_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4,)

    model = create_model(input_shape=(args.lattice_size, args.lattice_size, 1))
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=loss)   
    callbacks = define_callbacks(args)     

    if args.dump_model_summary:
        print_model_summary(model, args.stage_train_dir)

    history = model.fit(X_train, y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=args.epochs,
                        batch_size=args.batch_size)

    if args.save_model:
        model.save(make_path(args.stage_train_dir, 'saved-model_regression.h5'))

    if args.dump_history:
        write_numpy_dic_to_json(history.history, make_path(args.stage_train_dir, 'history_regression.json'))
    
    loss_test = model.evaluate(X_test, y_test, verbose=0)
    print('loss_test={:.3f}'.format(loss_test))

    return model, history

if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument("--odir", type=str, default='./saved_models/cnn-reg-test')
    parser.add_argument("--lattice_size", type=int, default=128)
    parser.add_argument("--dataset_size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--learning_rate", type=float, default=10e-4)
    
    parser.add_argument('--set_lr_scheduler', dest='set_lr_scheduler', action='store_true')
    parser.add_argument('--no-set_lr_scheduler', dest='set_lr_scheduler', action='store_false')
    parser.set_defaults(set_lr_scheduler=False)

    parser.add_argument('--set_checkpoint', dest='set_checkpoint', action='store_true')
    parser.add_argument('--no-set_checkpoint', dest='set_checkpoint', action='store_false')
    parser.set_defaults(set_checkpoint=False)

    parser.add_argument('--set_earlystopping', dest='set_earlystopping', action='store_true')
    parser.add_argument('--no-set_earlystopping', dest='set_earlystopping', action='store_false')
    parser.set_defaults(set_earlystopping=False)

    parser.add_argument('--set_tensorboard', dest='set_tensorboard', action='store_true')
    parser.add_argument('--no-set_tensorboard', dest='set_tensorboard', action='store_false')
    parser.set_defaults(set_tensorboard=False)

    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)

    parser.add_argument('--dump_model_summary', dest='dump_model_summary', action='store_true')
    parser.add_argument('--no-dump_model_summary', dest='dump_model_summary', action='store_false')
    parser.set_defaults(dump_model_summary=False)

    args = parser.parse_args()
    main(args)
