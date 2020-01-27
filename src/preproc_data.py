#!/usr/bin/env python

import os
import pandas as pd
from collections import Counter

#Define some useful functions.
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
    styles. This code remove paintings with incomplete information (such as
    art with missing labels) and subselect only those whose style is present
    in styles (set in master.py). The selected entries are stored in a .json
    reference file. A summary file is also created. 
    
    Parameters:
    -----------
    styles : ~list of strings
        List containing art styles to be incldued.
     
    Outputs:
    --------
    ./../output_files/preproc_train_info.json'
    ./../output_files/art_styles.dat'
    """    
    
    def __init__(self, styles):
        self.styles = styles        
        self.run_preproc_data()
    
    def load_data(self):
        """Load the csv file that conatins the filenames and respective styles.
        """
        inp_dir = './../input_data'
        attr_filepath = os.path.join(inp_dir, 'train_info.csv')
        self.attr = pd.read_csv(attr_filepath, header=0)

    def prepare_dataframe(self):
        """Trim the input dataframe to include only the filename and style
        columns and only the paintings that have the desired art styles and
        that were successfully resized.
        """
        #Remove columns that are not image name or art style. The filename
        #column can be 'new_filename' if using all_data_info.csv or 'filename'
        #if using train_info.csv.
        if 'new_filename' in self.attr.columns:
            self.attr = self.attr[['new_filename', 'style']]
            self.attr.rename(columns={'new_filename': 'filename'})
        else:
            self.attr = self.attr[['filename', 'style']]
                
        #Convert name of files to .png, which is their format after resizing.
        fnames = [f.split('.')[0] + '.png' for f in  self.attr.filename.values]
        self.attr['filename'] = fnames
        
        #Select only the paintings that were successfully resized.
        inp_dir = os.path.join('./../input_data/', 'train_grayscale')
        success_filenames = os.listdir(inp_dir) #These are .png files.
        self.attr = self.attr[self.attr['filename'].isin(success_filenames)]

    def clean_data(self):
        """Perform simple data cleaning. Remove rows whose style feature is
        nan. Also rename some art styles to remove non-alphabetic characters
        and group other art styles that are similar, such as early and high
        Renaissance. Creates a .dat file that summarises the number of
        paintings in each of the desired art-styles.
        """
        #Rename a few styles to remove special chars.
        #Combine a few styles together for better training statistics.
        conversor = {'Analytical\xa0Realism': 'Analytical-Realism',
                     'Sōsaku hanga': 'Sosaku hanga',
                     'Naïve Art (Primitivism)': 'Naive Art (Primitivism)',
                     'Early Renaissance': 'Renaissance',
                     'High Renaissance': 'Renaissance',
                     'Northern Renaissance': 'Renaissance',
                     'Mannerism (Late Renaissance)': 'Renaissance'}
        self.attr['style'] = self.attr['style'].replace(conversor)

        #Remove rows with 'nan' style.
        self.attr.dropna(axis=0, subset=['style'], inplace=True)
        display_info(self.attr, 'NaN removed')
        
        #Use only art styles in use_styles.
        self.attr = self.attr[self.attr['style'].isin(self.styles)]
        display_info(self.attr, 'Keep only certain art styles.')              

        #Write available styles that remain after preprocessing the data.
        write_available_styles(
          self.attr, os.path.join('./../output_data/', 'art_styles.dat'))    

    def write_output(self):
        """Create an output.json file using the processed dataframe of
        image filenames and respective styles.
        """
        fpath = os.path.join('./../output_data/', 'preproc_train_info.json')
        self.attr.to_json(fpath)

    def run_preproc_data(self):
        """Call all the routines above.
        """
        self.load_data()
        self.prepare_dataframe()
        self.clean_data()
        self.write_output()
