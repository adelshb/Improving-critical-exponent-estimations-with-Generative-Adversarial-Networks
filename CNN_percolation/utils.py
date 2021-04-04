
import os
import sys
import pandas as pd


def time_to_string(t):
    return t.strftime("%Y.%m.%d.%H.%M.%S")

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    path = os.path.realpath(path)
    return path



def write_numpy_dic_to_json(dic, path): 
    df = pd.DataFrame(dic) 
    with open(path, 'w') as f:
        df.to_json(f, indent=4, )




class Logger(object):
    """
    The following class prints the standard output to both console and file
    """
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        #self.console.flush()
        #self.file.flush()
        pass
       



