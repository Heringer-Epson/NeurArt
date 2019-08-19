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
    def __init__(self, img_x=400,img_y=600, n_style_min=100,
                 use_styles=None, gray=False):
        self.img_x = img_x
        self.img_y = img_y
        
        self.n_style_min = n_style_min
        
        #self.use_styles = use_styles
        #self.use_styles = ['Early Renaissance', 'Cubism', 'Ukiyo-e']
        self.use_styles = ['Impressionism', 'Renaissance', 'Cubism', 'Ukiyo-e',
                           'Dada', 'Pointillism']        
        self.top_dir = os.path.abspath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
