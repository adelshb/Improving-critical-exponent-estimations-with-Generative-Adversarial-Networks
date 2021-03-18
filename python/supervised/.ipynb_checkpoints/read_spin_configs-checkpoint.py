import glob
import os
import numpy as np
import sys
#=========================================================================
def read_files_convert_to_raw_array (
        fpaths, 
        lattice_size, 
        max_n_configs=-1
        ):
  
    if len(fpaths) == 0: 
        return np.empty(0)
    
    cumsum_nconfigs = 0
    data = []
    end_of_work = False
    for path in fpaths:
        fsize = os.path.getsize(path)
        nconfigs = fsize  // lattice_size
        cumsum_nconfigs += nconfigs
        
        if (max_n_configs > 0 and cumsum_nconfigs >= max_n_configs):
            canonic_nconfigs = nconfigs - (cumsum_nconfigs - max_n_configs)
            end_of_work = True
        else:
            canonic_nconfigs = nconfigs
                    
        data.append(np.fromfile(path, dtype=np.int8, 
                                count=canonic_nconfigs * lattice_size) )
        
        if end_of_work:
            break
    # end for path
        
    return np.concatenate(data).ravel()

#=========================================================================

def read_temp_dependent_files_then_merge(
        idir, L, T_arr, Tc, 
        max_configs_per_temperature=-1,
        base_name='spin',
        fmt='.bin',
        print_info_just_in_case=False
        ) :
    X, y = [], [] 
    i = 1
        
    for T in T_arr: 
        pattern = '{}(L={},T={:.4f})*{}'.format(base_name , L, T, fmt)
        fpaths = glob.glob(os.path.join(idir, pattern))
       
        XT = read_files_convert_to_raw_array(
            fpaths, L**2, max_configs_per_temperature)
        
        if XT.size == 0:
           continue
         
        #label = 0 if T < Tc else 1 if T > Tc else 2
        label = i if T != Tc else 0
        #label = 1 if T == Tc else 0
        
        
        yT = np.full(XT.size // L**2, label)
         
        X.append(XT)
        y.append(yT)
         
        if print_info_just_in_case:
            nT = XT.size // L**2
            print('T={}, n_configs={}'.format(T,nT) )
            print('files:', fpaths)
            print ('XT: ')
            for i in range(nT):
                print('label={}: '.format(yT[i]), (XT[i*L**2:(i+1)*L**2]+1)//2)
            print('-'*10)
            
        i += 1
        #
         
     # end for T
    
    X = np.array(X).reshape(-1, L, L, 1)
    y = np.array(y).reshape(-1, )
    return X, y
    


########################################################################

if __name__ == '__main__':
    idir = r'/Users/matthieu.sarkis/ml/research/criticality_data/config-files/L=256'
    T_arr = [2.25, 2.26, 2.2692, 2.28, 2.29]
    X, y = read_temp_dependent_files_then_merge(idir, 256, T_arr, 2.2692, max_configs_per_temperature=1)
    
    print(y)
    
    
    
    
    