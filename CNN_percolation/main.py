from argparse import ArgumentParser
import os, sys
import numpy as np
import train
import utils
from datetime import datetime
import percolation
import tensorflow as tf
#====================================================================
def prepare_data(**kwargs):

    p_down, p_up, p_increment = kwargs['p_down'], kwargs['p_up'], kwargs['p_increment']
    round_digit = kwargs['round_digit']
    L, n_configs_per_p = kwargs['L'], kwargs['n_configs_per_p']

    p_arr = np.round(np.arange(p_down, p_up, p_increment), round_digit)
    X, y, unique_labels = percolation.generate_data(L, p_arr, n_configs_per_p)
    
    return X, y, unique_labels
#====================================================================
def prepare_dirs(odir=None, folder_name=''):
    if odir is None: 
        odir= os.path.join(os.path.expanduser("~"), 'models')
   
    stage_train_dir = os.path.join(odir, folder_name) 
    os.makedirs(stage_train_dir, exist_ok=True) 
    return stage_train_dir
#====================================================================

def main(args):
    start_time = datetime.now()

    # init seed
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    

    X, y, unique_labels = prepare_data(L=args.L, 
                                       round_digit=args.round_digit, 
                                       p_down=args.p_down, 
                                       p_up=args.p_up, 
                                       p_increment=args.p_increment,
                                       n_configs_per_p=args.n_configs_per_p,
                                      )    

    stage_train_dir = prepare_dirs(folder_name='perc--' + utils.time_to_string(start_time))


    if True:
        print('# L={}'.format(args.L))
        print('# p_down={}, p_up={}, p_increment={}'.format(args.p_down, args.p_up, args.p_increment))
        print('# n_configs_per_p={}'.format(args.n_configs_per_p))
        print('# X.shape={} y.shape={}'.format(X.shape, y.shape))
        print('# labels.size={}'.format(len(unique_labels)))
        print('# stage_train_dir:', stage_train_dir)

    

    # %%
    # now train it
    model, history = train.train(X, y, 
                                 stage_train_dir=stage_train_dir, 
                                 random_state=args.random_state,
                                 patience=args.patience,
                                 test_size=args.test_size,
                                 epochs=args.epochs,
                                )



    # we have reached to the end!
    end_time = datetime.now()

    print ('# start_time={} end_time={} elpased_time={}'.\
    format(utils.time_to_string(start_time), 
           utils.time_to_string(end_time), 
           end_time - start_time) )


#====================================================================


if __name__ == '__main__':
    

    parser = ArgumentParser()

    
    # Model Parameters
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--round_digit", type=int, default=2)
    parser.add_argument("--n_configs_per_p", type=int, default=100)

    parser.add_argument("--p_down", type=float, default=0.50)
    parser.add_argument("--p_up", type=float, default=0.70)
    parser.add_argument("--p_increment", type=float, default=0.01)

    parser.add_argument("--random_state", type=int, default=42)
    
    
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    
   
    args = parser.parse_args()
    main(args)