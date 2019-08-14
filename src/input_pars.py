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
    def __init__(self, img_size=400, n_train=2201, n_style_min=None,
                 use_styles=None, gray=False):
        self.img_size = img_size
        self.n_train = n_train
        
        self.n_style_min = n_style_min
        #self.n_style_min = 1000
        
        
        #self.use_styles = use_styles
        #self.use_styles = ['Early Renaissance', 'Cubism', 'Impressionism']
        self.use_styles = ['Early Renaissance', 'Cubism']

        self.gray = True
        
        self.top_dir = os.path.abspath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
