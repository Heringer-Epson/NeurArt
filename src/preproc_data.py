#!/usr/bin/env python

import os
import pandas as pd
from collections import Counter

def display_info(df, setname):
    print ('\n\nStatus in %s' %(setname))
    print ('  number of art entries: %d' %(len(df.index)))
    print ('  number of unique art styles: %d' %(len(df['style'].unique())))

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
    
    def __init__(self, _inp):
        self._inp = _inp
        
        self.all_styles = None
        self.avail_styles = None
        
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
        if self._inp.use_styles is None:
            self.all_styles = self.attr['style'].unique()
            style_counter = Counter(self.attr['style'].values)
            relev_styles = [style for style in style_counter.keys()
                            if style_counter[style] >= self._inp.n_style_min]
            self.attr = self.attr[self.attr['style'].isin(relev_styles)]
            display_info(self.attr, 'Remove underrepresented art styles')

        else:
            self.attr = self.attr[self.attr['style'].isin(self._inp.use_styles)]
            display_info(self.attr, 'Keep only certain art styles.')            

        #Rename analytical.
        conversor = {'Analytical\xa0Realism': 'Analytical-Realism',
                     'Sōsaku hanga': 'Sosaku hanga',
                     'Naïve Art (Primitivism)': 'Naive Art (Primitivism)'}
        self.attr['style'] = self.attr['style'].replace(conversor)
        display_info(self.attr, 'Rename analytical')
        self.avail_styles = self.attr['style'].unique()

    def write_available_styles(self):
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'art_styles.dat')
        with open(fpath, 'w') as out:
            out.write('%s' %(self.avail_styles[0]))
            for style in self.avail_styles[1:]:  
                out.write('\n%s' %(style))    

    def write_output(self):
        #Store preprocessed dataframe.
        out_dir = os.path.join(self._inp.top_dir, 'output_data')
        fpath = os.path.join(out_dir, 'preproc_train_info.json')
        self.attr.to_json(fpath)

    def run_preproc_data(self):
        self.load_data()
        self.clean_data()
        self.write_available_styles()
        self.write_output()

    

    
