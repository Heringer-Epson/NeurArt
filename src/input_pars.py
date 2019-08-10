import os
import numpy as np

class Inp_Pars(object):
    """
    Description:
    ------------
    Define a set of input parameters to use to make the Dcolor vs SN rate plot
    in the class below. Detailed description of the parameters to follow.

    Parameters:
    -----------
    """
    def __init__(self, img_size, n_train=-1, n_style_min=100):
        self.img_size = img_size
        self.n_train = n_train
        self.n_style_min = n_style_min
        
        self.top_dir = os.path.abspath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
