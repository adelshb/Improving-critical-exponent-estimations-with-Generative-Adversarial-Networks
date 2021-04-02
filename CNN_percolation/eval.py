
from tensorflow import keras
import os, sys
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from main import get_data_set
#====================================================================
def evaluate(model_path, history_path=''):


    if os.path.exists(model_path):

        model = keras.models.load_model(model_path)

        # generate test set
        X, y, unique_labels = get_data_set(L=128, 
                                        round_digit=2, 
                                        p_down=0.58, 
                                        p_up=0.62, 
                                        p_increment=0.01,
                                        n_configs_per_p=100,)    

        print(unique_labels)    

        p_test = model.predict(X).argmax(axis=1)
        print(p_test)
        #conf_matrix = confusion_matrix(y, p_test)
        

        #plt.figure(figsize=(7,7))
        #sns.heatmap(conf_matrix, cmap='Blues', annot=True,
        #        xticklabels=unique_labels, yticklabels=unique_labels)
        #plt.xlabel('predicted')
        #plt.ylabel('true label')
        #plt.show()
    
    else:
        print ('# There is no model_path:', model_path)


#====================================================================

if __name__ == '__main__':
    
    evaluate(r'C:\Users\Ebi\Dropbox\Adel-Ebi-Matthieu\my-test\7\saved-files\2021.04.02.12.56.54\saved-model.h5')
    