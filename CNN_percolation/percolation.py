import numpy as np
import math

def percolation_configuration(L, p):
    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return 2 * spin - 1

def generate_data(L, p_arr, max_configs_per_p=1000):
    X, y = [], []
    unique_labels = []
  
    j = 0
    for p in p_arr:
        unique_labels.append(j)
        for i in range(max_configs_per_p):
            X.append(percolation_configuration(L, p))
            y.append(j)
        j += 1
    X = np.array(X).reshape(-1, L, L, 1)
    y = np.array(y).reshape(-1, )
    return X, y, unique_labels