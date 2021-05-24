
import os
import json
import pandas as pd


def time_to_string(t):
    return t.strftime("%Y.%m.%d.%H.%M.%S")

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    #path = os.path.realpath(path)
    return path



def write_numpy_dic_to_json(dic, path): 
    df = pd.DataFrame(dic) 
    with open(path, 'w') as f:
        df.to_json(f, indent=4, )

def print_model_summary(model, path):

    model.summary(print_fn=print)

    with open(make_path(path, 'model_summary.log'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    # print optimizations
    with open(make_path(path, 'optimizer.json'), 'w') as f:
        json.dump(model.optimizer.get_config(), f, indent=4, sort_keys=True)