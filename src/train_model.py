#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import Activation, MaxPooling2D, ZeroPadding2D
from keras_preprocessing.image import ImageDataGenerator
#from tensorflow.keras import regularizers, optimizers

#https://www.youtube.com/watch?v=cAICT4Al5Ow
#https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

class Train_Model(object):
    
    def __init__(self, _inp):
        self._inp = _inp
        self.proc_df = None
        self.X, self.y = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        
        if self._inp.gray:
            self.Nd = 1
        else:
            self.Nd = 3
        
        self.run_training()

    def read_processed_data(self):
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'proc_data.json')
        self.proc_df = pd.read_json(fpath)
                
        self.y = self.proc_df['style'].values
        #self.X = self.proc_df['data'].values.tolist()
        print (Counter(self.y))

        #Reshape arguments: (number of rows, size, size, layers (colors).
        
        #self.X = np.array(self.X).reshape(
        #  -1, self._inp.img_size, self._inp.img_size, self.Nd)

        #print (self.X.shape)
        
    def build_model(self):
        
        #Set the number of classes (in this case, how many different styles
        #are possible.
        n_classes = len(set(self.y))

        #Initialize model.
        model = Sequential()
        
        #Add first layer. Always a convolution.
        model.add(Conv2D(30, kernel_size=(3, 3),
                  activation='relu',
                  strides=2,
                  input_shape=(self._inp.img_size, self._inp.img_size, self.Nd)))
        #model.add(Dropout(0.5))             
        model.add(MaxPooling2D(pool_size=(2,2)))

        #Add additional layers.
        model.add(Conv2D(30, kernel_size=(3, 3),
                  strides=2,
                  activation='relu'))
        #model.add(Dropout(0.5))             
        model.add(MaxPooling2D(pool_size=(2,2)))        
        model.add(Conv2D(64, kernel_size=(3, 3),
                  strides=2,
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
                  epochs=3,
                  validation_split=0.2)

        #Save weights so that we can use the trained model without
        #training it in every run.
        N = str(self.X.shape[0]) #Number of images used to train model.
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        model.save_weights(os.path.join(out_dir, 'test_' + N + '.h5'))

    def build_model_2(self):
        n_classes = len(set(self.y))


        datagen=ImageDataGenerator(rescale=1./255., validation_split=0.20)

        img_dir = os.path.join(self._inp.top_dir, 'input_data/train')
        
        train_generator=datagen.flow_from_dataframe(
          dataframe=self.proc_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='training',
          batch_size=50,
          seed=42,
          shuffle=True,
          class_mode='categorical',
          target_size=(self._inp.img_size,self._inp.img_size))

        valid_generator=datagen.flow_from_dataframe(
          dataframe=self.proc_df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='validation',
          batch_size=50,
          seed=42,
          shuffle=True,
          class_mode='categorical',
          target_size=(self._inp.img_size,self._inp.img_size))

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(self._inp.img_size,self._inp.img_size,3)))
        model.add(Activation('relu'))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        #model.add(Conv2D(64, (3, 3), padding='same'))
        #model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #Add last layer.
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64))             
        model.add(Activation('relu'))
        model.add(Dropout(0.5))             
        model.add(Dense(n_classes, activation='softmax'))        
        
        #model.add(Dropout(0.25))
        #model.add(Flatten())
        #model.add(Dense(512))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        #model.add(Dense(n_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])        
        #model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
        #              loss='categorical_crossentropy',metrics=['accuracy'])

        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
        #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=1)

                    
    def run_training(self):
        self.read_processed_data()
        #self.build_model()
        self.build_model_2()
        
