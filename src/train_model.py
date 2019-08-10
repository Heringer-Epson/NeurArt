#!/usr/bin/env python

import os
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import Activation, MaxPooling2D, ZeroPadding2D

#https://www.youtube.com/watch?v=cAICT4Al5Ow

class Train_Model(object):
    
    def __init__(self, _inp):
        self._inp = _inp
        self.X, self.y = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        
        self.run_training()

    def read_processed_data(self):
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        
        pickle_X = open(os.path.join(out_dir, 'X.pickle'), 'rb')
        self.X = pickle.load(pickle_X)
        pickle_X.close()

        pickle_y = open(os.path.join(out_dir, 'y.pickle'), 'rb')
        self.y = pickle.load(pickle_y)
        pickle_y.close()
        
    def build_model(self):

        #Set the number of classes (in this case, how many different styles
        #are possible.
        n_classes = len(set(self.y))

        #Initialize model.
        model = Sequential()
        
        #Add first layer. Always a convolution.
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(self._inp.img_size, self._inp.img_size, 3)))
        model.add(MaxPooling2D(pool_size=(2,2)))

        #Add additional layers.
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))        
        model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2))) 
        
        #Add last layer.
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))             
        model.add(Dropout(0.5))             
        model.add(Dense(n_classes, activation='softmax'))
        
        #Compile model.
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        lb = LabelEncoder()
        y_enc = lb.fit_transform(self.y)
                      
        #Fit model.
        model.fit(self.X, y_enc,
                  batch_size=200,
                  epochs=4,
                  validation_split=0.2)

        #Save weights so that we can use the trained model without
        #training it in every run.
        N = str(self.X.shape[0]) #Number of images used to train model.
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        model.save_weights(os.path.join(out_dir, 'test_' + N + '.h5'))
            
    def run_training(self):
        self.read_processed_data()
        self.build_model()
        
