
import os
import sys


def time_to_string(t):
    return t.strftime("%Y.%m.%d.%H.%M.%S")

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    path = os.path.realpath(path)
    return path

def get_model_summary(model, print_fn=None):
    model.summary(print_fn=print_fn)




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
       



