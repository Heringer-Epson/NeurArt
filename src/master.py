#!/usr/bin/env python

import os
#Import NEURART routines. 
from preproc_data import Preproc_Data
from train_model import Train_Model
from apply_model import Apply_Model

class Master(object):
    """
    Code description: train a neural-network in art styles and make prediction.
    I follow the tutorial by sentdex: https://www.youtube.com/watch?v=j-3vuBynnOE
    """
    def __init__(self, train_flag, apply_flag, img_x, img_y, styles):
               
        if train_flag:
            Preproc_Data(styles)
            Train_Model(img_x, img_y)

        if apply_flag:
            Apply_Model(img_x, img_y, styles, top_dir)
       
if __name__ == '__main__':
    use_styles = ['Ukiyo-e', 'Cubism']
    #use_styles = ['Renaissance', 'Cubism', 'Ukiyo-e']
    Master(train_flag=True, apply_flag=True, img_x=400, img_y=600,
           styles=use_styles)


