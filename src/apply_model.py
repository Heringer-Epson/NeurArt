#!/usr/bin/env python

import os
import scipy
import json
import pandas as pd
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import load_model
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

class Apply_Model(object):
    
    def __init__(self, _inp):
        self._inp = _inp
        self.form_df = None
        self.model = None
        self.pred = None
        
        self.run_application()
        
    def load_input_data(self):
        
        out_fname, out_data = [], []
        
        inp_dir = os.path.join(self._inp.top_dir, 'input_images')
        for fname in os.listdir(inp_dir):

            img_data = load_img(os.path.join(inp_dir, fname))                                  
            dim = img_to_array(img_data).shape
            r = dim[0] / dim[1] #note that dim[0] is y and dim[1] is x.

            #Only include paintings that have a standard ratio.
            if ((r > 0.6 and r < 0.9) or (r > 1.10 and r < 1.67)):                
                #Most paintings are vertical and have a ratio of 4/3. If the
                #painting is horizontal, rotate it.
                if r < 1:
                    img = scipy.ndimage.rotate(img_data, 90, reshape=True)
                else:
                    img = scipy.ndimage.rotate(img_data, 0, reshape=True)
            
                #Convert image array to an Image object and resize it.
                img = Image.fromarray(img)
                #img = img.resize((900,1200), Image.ANTIALIAS)
                img = img.resize((200,300), Image.ANTIALIAS)
                out_fname.append(fname)
                out_data.append(np.array(img))
                plt.imshow(img)
                plt.show()

                del img #Delete variables to free memory.
            del img_data
        
        self.form_df = pd.DataFrame.from_dict({'filename': out_fname,
                                               'data': out_data})
    
    def build_pretrained_model(self):
        inp_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(inp_dir, 'model.h5')
        self.model = tf.keras.models.load_model(fpath)

    def apply_model_to_input(self):
        X = [img_array / 255. for img_array in  self.form_df['data'].values]
        print (np.asarray(X).shape)
        #print (X, len(X), type(X[0]))
        #print (np.array(X).shape)
        self.pred = self.model.predict(np.asarray(X), batch_size=1, steps=1)
        print (self.pred)

    def run_application(self):
        self.load_input_data()
        self.build_pretrained_model()
        self.apply_model_to_input()
