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
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # for ignoring the some of tf warnings
import tensorflow as tf

from src.CNN_regression.design_regression import create_model
from src.CNN_regression.utils import *

def main(args):

    X_train, X_test, y_train, y_test = generate_data(args.dataset_size, args.lattice_size)
    
    save_dir = os.path.join(args.odir, time_to_string(datetime.now()))
    os.makedirs(save_dir, exist_ok=True)

    with open(make_path(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4,)

    model = create_model(input_shape=(args.lattice_size, args.lattice_size, 1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)   
    callbacks = define_callbacks(args)     

    if args.dump_model_summary:
        print_model_summary(model, args.save_dir)

    history = model.fit(X_train, 
                        y_train,  
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks,
                        epochs=args.epochs,
                        batch_size=args.batch_size)

    if args.save_model:
        model.save(os.path.join(save_dir, 'saved-model_regression.h5'))

    if args.dump_history:
        write_numpy_dic_to_json(history.history, os.path.join(save_dir, 'history_regression.json'))

if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument("--odir", type=str, default='./saved_models/cnn_regression')
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

    parser.add_argument('--dump_history', dest='dump_history', action='store_true')
    parser.add_argument('--no-dump_history', dest='dump_history', action='store_false')
    parser.set_defaults(dump_history=True)

    args = parser.parse_args()
    main(args)
