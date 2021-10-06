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

from src.CNN_percolation import train_regression
from src.CNN_percolation.utils import make_path, time_to_string
from src.statphy.models.percolation import percolation_configuration

def main(args, print_args=True):
    
    start_time = datetime.now()

    X = []
    y = []
    for __ in range(args.train_size):
        y.append(np.random.rand())
        X.append(percolation_configuration(args.L, y[-1]))
    X=np.array(X)
    y=np.array(y)

    stage_train_dir = make_path(args.odir, time_to_string(start_time)) 
    os.makedirs(stage_train_dir, exist_ok=True)

    with open(make_path(stage_train_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4,)
    
    model, history = train_regression.train(X, y, 
                                            stage_train_dir=stage_train_dir, 
                                            epochs=args.epochs,
                                            batch_size=args.batch_size,    
                                            dropout_rate=args.dropout_rate,
                                            )

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--odir", type=str, default='./saved_models/cnn-reg-test')
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--train_size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", action='store', type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0)
   
    args = parser.parse_args()
    main(args)
