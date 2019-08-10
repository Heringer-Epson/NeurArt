#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from collections import Counter

def display_info(df, setname):
    print ('\n\nStatus in %s' %(setname))
    print ('  number of art entries: %d' %(len(df.index)))
    print ('  number of unique art styles: %d' %(len(df['style'].unique())))

class Preproc_Data(object):
    """
    The original data contains roughly 50,000 artworks and more than 120 art
    styles. About half of the styles is poorly represented, with less than 100
    paintings.
    This code trims the data to remove these poorly represented styles, with
    the threshold set by the n_style_min parameter (see  theinput_pars.py file). 
    Works that do not have an assigned style are also removed. Finally, the
    data are converted to a more ML friendly style and stored in json files.
    
    Parameters:
    -----------
    _inp : ~instance
        Instance of the Inp_Pars class defined in input_pars.py.
     
    Outputs:
    --------
    ./../output_files/X'
    """    
    
    def __init__(self, _inp):
        self._inp = _inp
        
        self.all_styles = None
        
        self.run_preproc_data()
    
    def load_data(self):
        inp_dir = os.path.join(self._inp.top_dir, 'input_data')
        
        #Read and store csv file containing attributes of each of the images.
        attr_filepath = os.path.join(inp_dir, 'train_info.csv')
        self.attr = pd.read_csv(attr_filepath, header=0)
        display_info(self.attr, 'Raw data')
        
        #Drop all columns except for filename and style.
        drop_cols = ['artist', 'title', 'genre', 'date']
        self.attr.drop(drop_cols, axis=1, inplace=True)
         
    def clean_data(self):

        #Remove rows with 'nan' style.
        self.attr.dropna(axis=0, subset=['style'], inplace=True)
        display_info(self.attr, 'NaN removed')
        
        #Remove underrepresented art styles.
        self.all_styles = self.attr['style'].unique()
        style_counter = Counter(self.attr['style'].values)
        relev_styles = [style for style in style_counter.keys()
                        if style_counter[style] >= self._inp.n_style_min]
        self.attr = self.attr[self.attr['style'].isin(relev_styles)]
        display_info(self.attr, 'Remove underrepresented art styles')

    def check_corrupted_files(self):
        #Not yet implemented.
        pass

    def write_output(self):
        """Store X and y. This avoids processing the data for every run.
        """
        out_dir = os.path.join(self._inp.top_dir, 'output_data')

    def run_preproc_data(self):
        self.load_data()
        self.clean_data()
        self.check_corrupted_files()
        #self.write_output()

    

    
