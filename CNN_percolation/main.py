from argparse import ArgumentParser
import os, sys
import numpy as np
import train
import utils
from datetime import datetime
import percolation
import tensorflow as tf
import pickle
#====================================================================
def get_data_set(**kwargs):

    p_down, p_up, p_increment = kwargs['p_down'], kwargs['p_up'], kwargs['p_increment']
    round_digit = kwargs['round_digit']
    L, n_configs_per_p = kwargs['L'], kwargs['n_configs_per_p']

    p_arr = np.round(np.arange(p_down, p_up+1e-10, p_increment), round_digit)
    X, y, unique_labels = percolation.generate_data(L, p_arr, n_configs_per_p)
    
    return X, y, unique_labels
#====================================================================
def get_dirs(odir='saved-files', folder_name=''): 
    stage_train_dir = os.path.join(odir, folder_name) 
    os.makedirs(stage_train_dir, exist_ok=True) 
    return stage_train_dir
#====================================================================

def main(args, print_args=True):
    start_time = datetime.now()

   

    # init seed
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    

    X, y, unique_labels = get_data_set(L=args.L, 
                                       round_digit=args.round_digit, 
                                       p_down=args.p_down, 
                                       p_up=args.p_up, 
                                       p_increment=args.p_increment,
                                       n_configs_per_p=args.n_configs_per_p,
                                      )    

    stage_train_dir = get_dirs(odir=args.odir, 
                               folder_name=utils.time_to_string(start_time))


    
    if print_args:
        print(72*'=')
        vargs = vars(args)
        for key in vargs:
            print('# {} = {}'.format(key, vargs[key]))
        
        print('# number_of_labels = {}'.format(len(unique_labels)))
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
    format(utils.time_to_string(start_time), 
           utils.time_to_string(end_time), 
           end_time - start_time) )


#====================================================================


if __name__ == '__main__':
    
    parser = ArgumentParser()

    
    # Model Parameters
    parser.add_argument("--odir", type=str, default='saved-files')
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--n_configs_per_p", type=int, default=100)

    parser.add_argument("--p_down", type=float, default=0.59)
    parser.add_argument("--p_up", type=float, default=0.60)
    parser.add_argument("--p_increment", type=float, default=0.01)
    parser.add_argument("--round_digit", type=int, default=2)

    parser.add_argument("--random_state", action='store', type=int, default=None)
    
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", action='store', type=int, default=None)
    
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    
    

    parser.add_argument('--n_gpus', type=int, default=1)
    
   
    args = parser.parse_args()
    main(args)