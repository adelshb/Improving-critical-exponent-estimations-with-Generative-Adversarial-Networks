
import os


def time_to_string(t):
    return t.strftime("%Y.%m.%d.%H.%M.%S")

def make_path(*paths):
    path = os.path.join(*[str(path) for path in paths])
    path = os.path.realpath(path)
    return path

def get_model_summary(model, print_fn=None):
    model.summary(print_fn=print_fn)
