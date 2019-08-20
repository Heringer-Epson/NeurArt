#!/usr/bin/env python

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

#Import NEURART routines. 
from input_pars import Inp_Pars
from preproc_data import Preproc_Data
from train_model import Train_Model
from apply_model import Apply_Model

class Master(object):
    """
    Code description: train a neural-network in art styles and produce prediction.
    I follow the tutorial by sentdex: https://www.youtube.com/watch?v=j-3vuBynnOE
    """
    def __init__(self, train_flag, apply_flag):
        self.train_flag = train_flag
        self.apply_flag = apply_flag
        self.inputs = None
    
    def run_master(self):
        self.inputs = Inp_Pars()
        
        if self.train_flag:
            Preproc_Data(self.inputs)
            Train_Model(self.inputs)
            pass
    
        if self.apply_flag:
            Apply_Model(self.inputs)
       


if __name__ == '__main__':
    Master(train_flag=True, apply_flag=True).run_master() 


