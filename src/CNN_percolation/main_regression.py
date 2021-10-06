# -*- coding: utf-8 -*-
#
# Written by Adel Sohbi https://github.com/adelshb.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
from argparse import ArgumentParser
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.CNN_percolation import train
from src.CNN_percolation.utils import make_path, time_to_string
from src.statphy.models.percolation import percolation_configuration

def main(args, print_args=True):
    
    start_time = datetime.now()

    # # init seed
    # np.random.seed(args.random_state)
    # tf.random.set_seed(args.random_state)
    
    X = []
    y = []
    for __ in range(args.train_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(args.L, y[-1]))
    X=np.array(X)
    y=np.array(y)

    # create the stage directory
    stage_train_dir = make_path(args.odir, time_to_string(start_time)) 
    os.makedirs(stage_train_dir, exist_ok=True)


    # # save unique_lables in stage_train_dir
    # with open(make_path(stage_train_dir, 'labels.json'), 'w') as f:
    #     json.dump(unique_labels, f, indent=4,)

    # save vars in stage_train_dir
    with open(make_path(stage_train_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4,)
    
    # now train the model
    model, history = train.train(X, y, 
                                 stage_train_dir=stage_train_dir, 
                                #  random_state=args.random_state,
                                #  patience=args.patience,
                                #  test_size=args.test_size,
                                 epochs=args.epochs,
                                 batch_size=args.batch_size,    
                                 dropout_rate=args.dropout_rate,
                                )

    # # we have reached to the end!
    # end_time = datetime.now()
    # print(65*'=')
    # print ('# start_time={} end_time={} elpased_time={}'.\
    # format(time_to_string(start_time), 
    #        time_to_string(end_time), 
    #        end_time - start_time) )

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--odir", type=str, default='./saved_models/cnn-reg-test')
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=120)
    # parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", action='store', type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0)
   
    args = parser.parse_args()
    main(args)
