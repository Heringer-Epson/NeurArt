#!/usr/bin/env python

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Process_Data(object):
    
    def __init__(self, _inp):
        self._inp = _inp
        self.attr = None
        self.img_df, self.final_df = None, None
        self.X, self.y = None, None
        
        self.run_process_data()
    
    def load_data(self):
        #Read and store csv file containing attributes of each of the images.
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        self.attr = pd.read_json(fpath)

        #Read each of the images and store info in lists.
        aux_data, aux_name = [], []
        inp_dir = os.path.join(self._inp.top_dir, 'input_data')
        images_filepath = os.path.join(inp_dir, 'train')

        fname_list = random.sample(list(self.attr['filename'].values),
                                   self._inp.n_train)
        for fname in fname_list:
            try:
                #print (fname)
                if self._inp.gray:
                    img_data = load_img(
                      os.path.join(images_filepath, fname),
                      color_mode='grayscale',
                      target_size=(self._inp.img_size, self._inp.img_size))
                else:
                    img_data = load_img(
                      os.path.join(images_filepath, fname),
                      target_size=(self._inp.img_size, self._inp.img_size))                    
                aux_data.append(img_to_array(img_data))
                aux_name.append(fname)
                #TEMP: Inspect images.
                #plt.imshow(img_data)
                #plt.show()
                del img_data #Delete variable to free memory.
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
        self.final_df = pd.merge(self.img_df, self.attr, on='filename')

        #Normalize pixel intensity.
        self.final_df['data'] = self.final_df['data'].apply(lambda x: x / 255.)
        print (self.final_df.shape)
        #Drop filename column.
        #self.final_df.drop(['filename'], axis=1, inplace=True)
        self.final_df.drop(['data'], axis=1, inplace=True)

    def write_output(self):
        #Store X and y. This avoids processing the data for every run.
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'proc_data.json')
        self.final_df.to_json(fpath)

    def run_process_data(self):
        self.load_data()
        self.process_dataframe()
        self.write_output()

    

    
