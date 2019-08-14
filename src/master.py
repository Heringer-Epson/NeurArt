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
from process_data import Process_Data
from train_model import Train_Model

class Master(object):
    """
    Code description: train a neural-network in art styles and produce prediction.
    I follow the tutorial by sentdex: https://www.youtube.com/watch?v=j-3vuBynnOE
    """
    def __init__(self, data_flag, train_flag, pred_flag):
        self.data_flag = data_flag
        self.train_flag = train_flag
        self.pred_flag = pred_flag
        self.inputs = None
    
    def run_master(self):
        self.inputs = Inp_Pars()
        
        if self.data_flag:
            Preproc_Data(self.inputs)
            Process_Data(self.inputs)
            pass
        if self.train_flag:
            Train_Model(self.inputs)
            pass
        if self.pred_flag:
            pass            


if __name__ == '__main__':
    Master(
      data_flag=True, train_flag=True, pred_flag=False).run_master() 


