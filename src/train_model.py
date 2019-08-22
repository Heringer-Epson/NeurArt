#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import Activation, MaxPooling2D, ZeroPadding2D
from keras_preprocessing.image import ImageDataGenerator

class Train_Model(object):
    
    def __init__(self, img_x, img_y, top_dir):
        self.img_x, self.img_y = img_x, img_y
        self.top_dir = top_dir

        self.proc_df = None
        self.train_generator, self.valid_generator = None, None
        self.model, self.labels = None, None
        self.pred = None
        self.n_classes = None
        
        self.run_training()      

    def read_processed_data(self):
        out_dir = os.path.join(self.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        self.proc_df = pd.read_json(fpath)
        self.proc_df = self.proc_df.sample(frac=1)
        self.n_classes = len(set(self.proc_df['style']))

    def initialize_generators(self):
        #Note: target_size argument is (height,width).
        batch_size=50
        datagen=ImageDataGenerator(rescale=1./255.,
                                   validation_split=0.10,
                                   horizontal_flip=True)
        img_dir = os.path.join(self.top_dir, 'input_data/train_clean')
                
        self.train_generator=datagen.flow_from_dataframe(
          dataframe=self.proc_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='training',
          batch_size=batch_size,
          seed=10,
          shuffle=True,
          class_mode='categorical',
          target_size=(self.img_y,self.img_x))

        self.valid_generator=datagen.flow_from_dataframe(
          dataframe=self.proc_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='validation',
          batch_size=batch_size,
          seed=10,
          shuffle=True,
          class_mode='categorical',
          target_size=(self.img_y,self.img_x))

        self.labels = (self.train_generator.class_indices)
        
    def build_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', strides=2,
                         input_shape=(self.img_y,self.img_x,3)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(32, (3, 3), strides=2,))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(64, (3, 3), strides=2,))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #Add last layer.
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256))             
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))             
        self.model.add(Dense(self.n_classes, activation='softmax'))        

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])        

        step_train=self.train_generator.n//self.train_generator.batch_size
        step_valid=self.valid_generator.n//self.valid_generator.batch_size

        self.model.fit_generator(
          generator=self.train_generator,
          steps_per_epoch=step_train,
          validation_data=self.valid_generator,
          validation_steps=step_valid,
          epochs=1)
        
        self.model.evaluate_generator(generator=self.valid_generator,
        steps=step_valid)

        #Save weights so that we can use the trained self.model without
        #training it in every run.
        fpath = os.path.join(self.top_dir, 'output_data/model.h5')
        self.model.save(fpath)

    def run_training(self):
        self.read_processed_data()
        self.initialize_generators()
        self.build_model()
