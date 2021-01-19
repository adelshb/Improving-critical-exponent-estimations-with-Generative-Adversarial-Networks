import glob
import os
import numpy as np
import sys

def read_data(idir, L, T, max_n_configs=-1, base='spin', fmt='.bins'):
    pattern = os.path.join(idir, '{}(L={},T={:.4f})*{}'.format(base,L, T, fmt))
    flist = glob.glob(pattern)
    
    if len(flist) == 0:
        return np.empty(0)
    
    data = []
    for path in flist:
        data.append(np.fromfile(path, dtype='int8'))
        
    data = np.array(data).reshape(-1,L,L,1)
    
    if max_n_configs > 0:
        return data[:max_n_configs, :, :, :]
    return data

def merge_data(idir, L, T_arr, Tc, max_configs_per_temperature=-1):
    X = []
    y = []
    for T in T_arr:
        
        if T < Tc: 
            label = 0
        elif T > Tc:
            label = 1
        else:
            label = 2
            
        XT = read_data(idir, L, T, max_configs_per_temperature)
        if XT.size == 0:
            continue
        yT = np.full((XT.shape[0],), label)
        
        X.append(XT)
        y.append(yT)
    
    X = np.array(X).reshape(-1,L,L,1)
    y = np.array(y).reshape(-1,)
    
    return X, y