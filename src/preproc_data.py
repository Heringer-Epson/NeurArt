#!/usr/bin/env python

import os
import pandas as pd
from collections import Counter

def display_info(df, setname):
    print ('\n\nStatus in %s' %(setname))
    print ('  number of art entries: %d' %(len(df.index)))
    print ('  number of unique art styles: %d' %(len(df['style'].unique())))

def write_available_styles(df, fpath):
    style_counter = Counter(df['style'].values)
    with open(fpath, 'w') as out:
        out.write('Style: Count')
        for (style,count) in style_counter.most_common():  
            out.write('\n%s: %i' %(style,count))   

class Preproc_Data(object):
    """
    The original data contains roughly 80,000 artworks and more than 130 art
    styles. About half of the styles is poorly represented, with less than 100
    paintings.
    This code trims the data to remove these poorly represented styles, with
    the threshold set by the n_style_min parameter (see  theinput_pars.py file). 
    Works that do not have an assigned style are also removed.
    
    Parameters:
    -----------
    _inp : ~instance
        Instance of the Inp_Pars class defined in input_pars.py.
     
    Outputs:
    --------
    ./../output_files/X'
    """    
    
    def __init__(self, styles, top_dir):
        self.styles = styles
        self.top_dir = top_dir

        self.all_styles = None
        self.out_dir = None
        
        self.run_preproc_data()
    
    def load_data(self):
        inp_dir = os.path.join(self.top_dir, 'input_data')
        self.out_dir = os.path.join(self.top_dir, 'output_data')
        
        #Read and store csv file containing attributes of each of the images.
        #attr_filepath = os.path.join(inp_dir, 'train_info.csv')
        #self.attr = pd.read_csv(attr_filepath, header=0)
        attr_filepath = os.path.join(inp_dir, 'train_info_clean.json')
        self.attr = pd.read_json(attr_filepath)

    def clean_data(self):

        #Drop all columns except for filename and style.
        drop_cols = ['artist', 'title', 'genre', 'date']
        self.attr.drop(drop_cols, axis=1, inplace=True)

        #Rename a few style so that the strings have fewer special chars.
        conversor = {'Analytical\xa0Realism': 'Analytical-Realism',
                     'Sōsaku hanga': 'Sosaku hanga',
                     'Naïve Art (Primitivism)': 'Naive Art (Primitivism)',
                     'Early Renaissance': 'Renaissance',
                     'High Renaissance': 'Renaissance',
                     'Northern Renaissance': 'Renaissance',
                     'Mannerism (Late Renaissance)': 'Renaissance'}
        self.attr['style'] = self.attr['style'].replace(conversor)
        
        #Display information on raw dataset and write sorted file with styles.
        display_info(self.attr, 'Raw data')
        write_available_styles(
          self.attr, os.path.join(self.out_dir, 'art_styles_all.dat'))        

        #Remove rows with 'nan' style.
        self.attr.dropna(axis=0, subset=['style'], inplace=True)
        display_info(self.attr, 'NaN removed')
        
        #Use only art styles in use_styles.
        self.attr = self.attr[self.attr['style'].isin(self.styles)]
        display_info(self.attr, 'Keep only certain art styles.')              

        #Write available styles that remain after preprocessing the data.
        write_available_styles(
          self.attr, os.path.join(self.out_dir, 'art_styles.dat'))    

    def write_output(self):
        out_dir = os.path.join(self.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        self.attr.to_json(fpath)

    def run_preproc_data(self):
        self.load_data()
        self.clean_data()
        self.write_output()
