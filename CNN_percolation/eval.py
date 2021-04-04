
from tensorflow import keras
import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from main import get_data_set
import pandas as pd
import math
from argparse import ArgumentParser
#====================================================================
def get_proper_lables_and_y(path_labels_train, labels_test, y_test):
    """
    get the true (trained) labels from path_label_train.
    since the labels_test may not be consistant with the 
    trained lables, so we calculate the proper labels_test and
    propter y_test as well.
    """

    with open(path_labels_train, 'r') as f:
        labels_train = json.load(f)

    if labels_train == labels_test:
        return labels_train, labels_test, y_test


    labels_test_new = {}
    I = np.array([], dtype=int)
    for k in labels_test.keys():
        if k in labels_train:
            I = np.append(I, labels_train[k])
            labels_test_new[k] = labels_train[k]
        else:
            print ('# Error the labels_test and labels_train are not compatible')
            sys.exit()
    
    return labels_train, labels_test_new, I[y_test]

#====================================================================
def plot_history(history_path):

    if os.path.exists(history_path):
        dframe = pd.read_json(history_path)

        dfrme_loss = dframe[['loss', 'val_loss']]
        dfrme_accu = dframe[['accuracy', 'val_accuracy']]

        plt.figure(1)
        dfrme_loss.plot(xlabel='epoch')
        plt.savefig('loss.pdf')
        #plt.show()

        plt.figure(2)
        dfrme_accu.plot(xlabel='epoch')
        plt.xlabel('epoch')
        plt.savefig('accuracy.pdf')
        #plt.show()
    else:
        print ('# There is not history_path:', history_path)
        return

#====================================================================
def evaluate_model(model_path, labels_path, args_path):

    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
    else:
        print ('# There is not args_path:', args_path)
        return

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        print ('# There is not model_path:', model_path)
        return


    print(' --> args_of_trained_model:', args)

    #model_input_shape = model.layers[0].output_shape[0] #(None,L,L,1)
    #L = model_input_shape[1]

     # generate test set
    X, y, labels = get_data_set(L=args['L'], 
                                round_digit=args['round_digit'], 
                                p_down=args['p_down'], 
                                p_up=args['p_up'],
                                p_increment=args['p_increment'],
                                n_configs_per_p=500,)    

    if os.path.exists(labels_path):

        # notice that this y and lables may not be consistant 
        # with the lables and y derived from train set
        labels_train, labels_test, y_test = get_proper_lables_and_y(
            labels_path, labels, y)
        
        X_test = X.astype(np.float32)
        
    else:
        print ('# There is not labels_path:', labels_path)
        return


    # evaluate test set
    loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=1)
    print('loss_test={:.3f}, accuracy_test={:.3f}'.format(loss_test, accuracy_test))


    # predict test set
    y_pred = model.predict(X_test).argmax(axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred, 
                                   labels=np.arange(len(labels_train)))
    
    plt.figure(figsize=(7,7))
    sns.heatmap(conf_matrix, xticklabels=labels_train, yticklabels=labels_train, 
                cmap='Blues', annot=True,)
    plt.xlabel('predicted')
    plt.ylabel('true label')
    plt.savefig('conf_mat.pdf')
    #plt.show()
    
#====================================================================
def main(args, 
         fname_model = 'saved-model.h5',
         fname_label = 'labels.json',
         fname_args = 'args.json',
         fname_history = 'history.json',
        ):


    if os.path.exists(args.trained_dir):
        print('# trained_dir:', args.trained_dir)



        fpath_model = os.path.join(args.trained_dir, fname_model)
        fpath_label = os.path.join(args.trained_dir, fname_label)
        fpath_args = os.path.join(args.trained_dir, fname_args)
        fpath_history = os.path.join(args.trained_dir, fname_history)


        # part plot histoy
        print('# Plotting history of loss ...')
        plot_history(fpath_history)

        # part predict and evaluate
        print('# Evaluating and plotting confusion_matrix ...')
        evaluate_model(fpath_model, fpath_label, fpath_args)

    else:
        print('# Error: there is no trained_dir:', args.trained_dir)


#====================================================================

if __name__ == '__main__':

    parser = ArgumentParser()

    
    # Model Parameters
    parser.add_argument("--trained_dir", type=str, 
                        default="saved-files/2021.04.04.18.17.23")


    args = parser.parse_args()
    main(args)

    
    
  
