import os
from argparse import ArgumentParser
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

import src.CNN_percolation.train
from src.CNN_percolation.utils import make_path, time_to_string
from src.statphy.models import percolation

#====================================================================
def main(args, print_args=True):
    
    start_time = datetime.now()

    # init seed
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    
    
    # create the data set; X, y and labels
    p_arr = np.round(np.arange(args.p_down, args.p_up + 1e-10, args.p_increment), 
                     args.round_digit)
    X, y, unique_labels = percolation.generate_data(args.L, p_arr, args.n_configs_per_p)
    

    # create the stage directory
    stage_train_dir = make_path(args.odir, time_to_string(start_time)) 
    os.makedirs(stage_train_dir, exist_ok=True)


    # save unique_lables in stage_train_dir
    with open(make_path(stage_train_dir, 'labels.json'), 'w') as f:
        json.dump(unique_labels, f, indent=4,)

    
    vargs = vars(args)
    

    # save vars in stage_train_dir
    with open(make_path(stage_train_dir, 'args.json'), 'w') as f:
        json.dump(vargs, f, indent=4,)


    # print args
    if print_args:
        print(72*'=')
        
        for key in vargs:
            print('# {} = {}'.format(key, vargs[key]))
        
        print('# number_of_labels = {}'.format(len(labels)))
        print('# stage_train_dir = {}'.format(stage_train_dir))
        print('# X.shape={}  y.shape={}'.format(X.shape, y.shape))
        print(72*'=')

    

    # %%
    # now train the model
    model, history = train.train(X, y, 
                                 stage_train_dir=stage_train_dir, 
                                 random_state=args.random_state,
                                 patience=args.patience,
                                 test_size=args.test_size,
                                 epochs=args.epochs,
                                 n_gpus=args.n_gpus,
                                 batch_size=args.batch_size,    
                                 dropout_rate=args.dropout_rate,
                                )



    # we have reached to the end!
    end_time = datetime.now()
    print(65*'=')
    print ('# start_time={} end_time={} elpased_time={}'.\
    format(time_to_string(start_time), 
           time_to_string(end_time), 
           end_time - start_time) )


#====================================================================


if __name__ == '__main__':
    
    parser = ArgumentParser()

    
    # Model Parameters
    parser.add_argument("--odir", type=str, default='saved_files')
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--n_configs_per_p", type=int, default=10)

    parser.add_argument("--p_down", type=float, default=0.54)
    parser.add_argument("--p_up", type=float, default=0.62)
    parser.add_argument("--p_increment", type=float, default=0.02)
    parser.add_argument("--round_digit", type=int, default=2)

    parser.add_argument("--random_state", action='store', type=int, default=None)
    
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", action='store', type=int, default=32)
    
    parser.add_argument("--dropout_rate", type=float, default=0)
    
    

    parser.add_argument('--n_gpus', type=int, default=1)
    
   
    args = parser.parse_args()
    main(args)
