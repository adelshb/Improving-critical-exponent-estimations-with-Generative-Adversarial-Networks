import numpy as np

def percolation_configuration(L, p):
    spin = (np.random.random(size=(L,L)) < p).astype(np.int8)
    return 2 * spin - 1

def read_percolation_data(L, p_arr, pc, max_configs_per_p=2000):
    X = []
    y = []
    j = 1
    for p in p_arr:
        #label = j if p != pc else 0
        label = 1 if p == pc else 0
        for i in range(max_configs_per_p):
            X.append(percolation_configuration(L, p))
            y.append(label)
        j += 1
    X = np.array(X).reshape(-1, L, L, 1)
    y = np.array(y).reshape(-1, )
    return X, y