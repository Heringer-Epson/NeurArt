#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.layers import Activation, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator

#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

class Train_Model(object):
    """
    Description:
    ------------
    Trains a Convolutional Neural Network (CNN) model on a sample of painting
    images. The images are accessed via an ImageDataGenerator instance, such
    that only small batches are retrieved at once to prevent memory overload.
    The trained model is stored in an output file.

    Parameters:
    -----------
    img_x : ~in
        Number of pixels respective to the image width. Standard cropping
        perfomed in data pre-processing is 900. Set in master.py.
    img_y : ~in
        Number of pixels respective to the image height. Standard cropping
        perfomed in data pre-processing is 1200. Set in master.py.        

    Return:
    -------
    './../output_data/model.h5'
    """        
    def __init__(self, img_x, img_y):
        self.img_x, self.img_y = img_x, img_y

        #CNN model parameters.
        self.batch_size = 15 ##To increase when more computing power is availalbe.
        self.window = (3,2) #Preserve the image proportion (y=1200, x=900 pixels).
        self.pool = (3,2) #Get max in window of (y,x).
        self.epochs = 10

        self.df = None
        self.train_generator, self.valid_generator = None, None
        self.model, self.labels = None, None
        self.n_classes = None
        
        self.run_training()      

    def read_processed_data(self):
        """Read the dataframe containing the image filenames and their styles.
        """
        out_dir = './../output_data'
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        self.df = pd.read_json(fpath)
        
        #Number of classes being used.
        self.n_classes = len(set(self.df['style']))
        
    def initialize_generators(self):
        """Construct iterators that will collect a batch of row (images) from
        the master dataframe and retrieve the corresponding images from the
        directory specified.
        
        * Set color_mode='grayscale' to speed up calculations. Assume the color
        information is not that relevant.
        * Set shuffle=True so that the training/validation sets are not biased
        towards sequence of rows that all contain the same class. 
        * Fix a seed for both training and validation so that these datasets
        do not repeat images.
        * target_size is image (height,width). Passed directly from master.py.
        """
        datagen=ImageDataGenerator(rescale=1./255.,
                                   validation_split=0.20,
                                   horizontal_flip=True)
        img_dir = os.path.join('./../input_data/', 'train_grayscale')
                
        seed = 2314125
        self.train_generator=datagen.flow_from_dataframe(
          dataframe=self.df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='training',
          color_mode='grayscale',
          batch_size=self.batch_size,
          seed=seed,
          shuffle=True,
          class_mode='categorical',
          target_size=(self.img_y,self.img_x))

        self.valid_generator=datagen.flow_from_dataframe(
          dataframe=self.df,
          directory=img_dir,
          x_col='filename',
          y_col='style',
          subset='validation',
          color_mode='grayscale',
          batch_size=self.batch_size,
          seed=seed,
          shuffle=True,
          class_mode='categorical',
          target_size=(self.img_y,self.img_x))

        #Mapping between labels and encoded output.
        #E.g. {'Cubism': 0, 'Ukiyo-e': 1}.
        self.labels = (self.train_generator.class_indices)
        
    def build_model(self):
        """Assemble CNN layers. Here I use 5 layers in an attempt to capture
        complex features that may be useful in distinguishing between styles.
        
        * A conservative dropout 'layer' is used to prevent over-fitting.
        * The relu activation function is usually preferred as it is the
        usual choice to prevent vanishing gradients. Future implementations
        will also test leaky ELU.
        * pool and window sizes are somewhat arbitrary, but chosen as to
        preserve the ratio of the images (height=1200,width=900).
        * Loss function is the standard categorical cross-entropy. Other
        options have not yet been tested.
        * Optimizer is the standard 'Adam'. Other options have not yet been
        tested.
        """
        self.model = Sequential()
        self.model.add(Conv2D(64, self.window, padding='same', strides=2,
                       input_shape=(self.img_y,self.img_x,1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool))
        
        self.model.add(Conv2D(32, self.window, strides=2,))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool))
        self.model.add(Dropout(0.20))
        
        self.model.add(Conv2D(64, self.window, strides=2,))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        
        self.model.add(Conv2D(64, self.window, strides=2,))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool))
        self.model.add(Dropout(0.20))
        
        #Add last layer.
        self.model.add(Flatten()) #Flatten to 1D to feed a Dense layer.
        self.model.add(Dense(256))           
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.30))             
        self.model.add(Dense(self.n_classes, activation='softmax'))        

        #Compile model.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])        

    def train_cnn(self):
        """Train the CNN model using the image generators defined above.
        """
        
        #Defined train and test steps such that the number of steps is just
        #sufficient to cover all images once for the given batch size.
        step_train=self.train_generator.n//self.train_generator.batch_size
        step_valid=self.valid_generator.n//self.valid_generator.batch_size

        #Define an early stopping condition to prevent over-fitting for
        #large number of epochs. Stop the training if the validation accuracy
        #decreases for 5 epochs in a row.
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        #Fit model.
        self.model.fit_generator(
          generator=self.train_generator,
          steps_per_epoch=step_train,
          validation_data=self.valid_generator,
          validation_steps=step_valid,
          epochs=self.epochs,
          callbacks=[early_stopping])
        
        self.model.evaluate_generator(generator=self.valid_generator,
        steps=step_valid)

        #Save weights so that we can use the trained self.model without
        #training it in every run.
        fpath = os.path.join('./../output_data/', 'model.h5')
        self.model.save(fpath)

    def run_training(self):
        """Call all the routines above.
        """
        self.read_processed_data()
        self.initialize_generators()
        self.build_model()
        self.train_cnn()
