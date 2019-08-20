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
    
    def __init__(self, _inp):
        self._inp = _inp
        self.train_df, self.val_df = None, None 
        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.model = None
        self.pred = None
        self.n_classes = None      
        self.run_training()

    def read_processed_data(self):
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        proc_df = pd.read_json(fpath)
        proc_df = proc_df.sample(frac=1)
        self.train_df, self.test_df = train_test_split(proc_df, test_size=0.1)
        self.n_classes = len(set(self.train_df['style']))

    def initialize_generators(self):
        #Note: target_size argument is (height,width).
        batch_size=50
        datagen=ImageDataGenerator(rescale=1./255.,
                                   validation_split=0.20,
                                   #width_shift_range = 0.2,
                                   horizontal_flip=True)
        #img_dir = os.path.join(self._inp.top_dir, 'input_data/train')
        img_dir = os.path.join(self._inp.top_dir, 'input_data/train_clean')
                
        self.train_generator=datagen.flow_from_dataframe(
          dataframe=self.train_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='training',
          batch_size=batch_size,
          seed=10,
          shuffle=True,
          class_mode='categorical',
          target_size=(self._inp.img_y,self._inp.img_x))

        self.valid_generator=datagen.flow_from_dataframe(
          dataframe=self.train_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='validation',
          batch_size=batch_size,
          seed=10,
          shuffle=True,
          class_mode='categorical',
          target_size=(self._inp.img_y,self._inp.img_x))
        
        test_datagen=ImageDataGenerator(rescale=1./255.)
        self.test_generator=test_datagen.flow_from_dataframe(
          dataframe=self.test_df,
          directory=img_dir,
          x_col='filename',
          y_col=None,
          batch_size=batch_size,
          seed=10,
          shuffle=False,
          class_mode=None,
          target_size=(self._inp.img_y,self._inp.img_x))

    def build_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', strides=2,
                         input_shape=(self._inp.img_y,self._inp.img_x,3)))
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
        step_test=self.test_generator.n//self.test_generator.batch_size

        self.model.fit_generator(
          generator=self.train_generator,
          steps_per_epoch=step_train,
          validation_data=self.valid_generator,
          validation_steps=step_valid,
          epochs=2)
        
        self.model.evaluate_generator(generator=self.valid_generator,
        steps=step_test)

        self.test_generator.reset()
        self.pred=self.model.predict_generator(self.test_generator,
                                               steps=step_test)

    def write_output(self):
        
        predicted_class_indices=np.argmax(self.pred,axis=1)
        labels = (self.train_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]


        results=pd.DataFrame({'Filename': self.test_generator.filenames,
                              'True_style': self.test_df['style'].values,
                              'Predicted_style': predictions})

        aux_pred = np.transpose(self.pred)
        for i in range(self.n_classes):
            results[labels[i]] = aux_pred[i]       

        fpath = os.path.join(self._inp.top_dir, 'output_data/results.csv')
        results.to_csv(fpath,index=False,header=True)

        #Save weights so that we can use the trained self.model without
        #training it in every run.
        #fpath = os.path.join(self._inp.top_dir, 'output_data/weights.h5')
        #self.model.save_weights(fpath)
        fpath = os.path.join(self._inp.top_dir, 'output_data/model.h5')
        self.model.save(fpath)

    def run_training(self):
        self.read_processed_data()
        self.initialize_generators()
        self.build_model()
        self.write_output()
        
