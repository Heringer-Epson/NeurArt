#!/usr/bin/env python

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from collections import Counter

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 30.

class Make_Fig(object):
    """
    Description:
    ------------
    TBW.
  
    Parameters:
    -----------
    show_fig : ~bool
        True of False. Whether to show or not the produced figure.
    save_fig : ~bool
        True of False. Whether to save or not the produced figure.
     
    Outputs:
    --------
    ./../OUTPUT_FILES/output_figures/Fig_style_hist.pdf
    """           
    def __init__(self, show_fig, save_fig):
        self.show_fig = show_fig
        self.save_fig = save_fig

        self.top_dir = os.path.abspath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
        
        self.ax = None
        self.style_counter = None
        
        self.run_plot()

    def read_data(self):

        #Read and store csv file containing attributes of each of the images.
        inp_dir = os.path.join(self.top_dir, 'input_data')
        attr_filepath = os.path.join(inp_dir, 'train_info.csv')
        attr = pd.read_csv(attr_filepath, header=0)
                
        #Create dictionary containing a count of each style.
        self.style_counter = Counter(attr['style'].values)

    def plot_histogram(self):

        df = pd.DataFrame.from_dict(
          self.style_counter, orient='index', columns=['Count'])
        print (df)
        df = df.sort_values(by='Count', ascending=False)
        df['Count'] = df['Count'].apply(lambda x: np.log10(x))
        print (df)
        self.ax = df.plot(kind='bar', figsize=(15,9))

        self.ax.axhline(y=2., ls='--', color='k', lw=2.)
        self.ax.get_legend().remove()

    def set_fig_frame(self):

        self.ax.set_ylabel(r'$\mathrm{log\,(Count)}$', fontsize=fs)
        self.ax.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax.tick_params(axis='x', which='major', labelsize=10., pad=16)
        self.ax.tick_params(length=12, width=2., which='major')
        #self.ax.tick_params(length=6, width=2., which='minor')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.set_ylim(0.,4.)
        self.ax.yaxis.set_minor_locator(MultipleLocator(.5))
        self.ax.yaxis.set_major_locator(MultipleLocator(1.))

    def manage_output(self):
        plt.tight_layout()
        if self.save_fig:
            out_dir = os.path.join(self.top_dir, 'output_figures')
            fpath = os.path.join(out_dir, 'Fig_style_hist.pdf')
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.read_data()
        self.plot_histogram()
        self.set_fig_frame()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(show_fig=False, save_fig=True)
