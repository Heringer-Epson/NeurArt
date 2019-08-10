#!/usr/bin/env python

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Process_Data(object):
    
    def __init__(self, _inp):
        self._inp = _inp
        self.attr = None
        self.img_df = None
        self.X, self.y = None, None
        
        self.run_process_data()
    
    def load_data(self):
        inp_dir = os.path.join(self._inp.top_dir, 'input_data')
        
        #Read and store csv file containing attributes of each of the images.
        attr_filepath = os.path.join(inp_dir, 'train_info.csv')
        self.attr = pd.read_csv(attr_filepath, header=0)

        #Read each of the images and store info in lists.
        aux_data, aux_name = [], []
        images_filepath = os.path.join(inp_dir, 'train_1')

        for fname in os.listdir(images_filepath)[0:self._inp.n_train]:
            try:
                img_data = load_img(
                  os.path.join(images_filepath, fname),
                  target_size=(self._inp.img_size, self._inp.img_size))
                aux_data.append(img_to_array(img_data))
                aux_name.append(fname)
                del img_data #Delete variable to free memory.
                #TEMP: Inspect images.
                plt.imshow(img_to_array(img_data))
                plt.show()
            except:
                pass

        #Store images and filenames in a dataframe.
        self.img_df = pd.DataFrame({'data': aux_data, 'filename':aux_name})
        
    def process_dataframe(self):
        """Perform relevant operations to clean up the data and make it usable
        for ML routines.
        """
        #Combine the style, date, etx information in selt.attr with the
        #image data. Use the filename column as reference.
        final_df = pd.merge(self.img_df, self.attr, on='filename')

        #Drop data that is irrelevant for this project and clean rows (no NaN).
        final_df.drop(['filename', 'genre', 'date', 'artist', 'title'],
                      axis=1, inplace=True)
        final_df.dropna(axis=0, subset=['style', 'data'], inplace=True)

        #Create standard feature and target arrays for the ML implementation.
        self.y = final_df['style'].values
        self.X = list(final_df['data'].values) #As list to facilitate reshaping below.
        
        #Reshape arguments: (number of rows, size, size, layers (colors).
        self.X = np.array(self.X).reshape(
          -1, self._inp.img_size, self._inp.img_size, 3)
        self.X = self.X / 255. #Normalize the pixel information.
        
        print (self.X.shape)

    def write_output(self):
        """Store X and y. This avoids processing the data for every run.
        """
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        
        X_out = open(os.path.join(out_dir, 'X.pickle'), 'wb')
        pickle.dump(self.X, X_out)
        X_out.close()

        y_out = open(os.path.join(out_dir, 'y.pickle'), 'wb')
        pickle.dump(self.y, y_out)
        y_out.close()        

    def run_process_data(self):
        self.load_data()
        self.process_dataframe()
        self.write_output()

    

    
