#!/usr/bin/env python

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
fs = 24.

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
        self.ratio = []
        self.ratio_clean = []

        self.top_dir = os.path.abspath(
          os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
          
        fig = plt.figure(figsize=(12,16))
        self.ax1 = fig.add_subplot(211)
        self.ax2 = fig.add_subplot(212)
        
        self.run_plot()

    def set_fig_frame(self):
        self.ax1.set_ylabel(r'$\mathrm{log\,(Count)}$', fontsize=fs)
        self.ax1.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax1.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax1.tick_params(length=12, width=2., which='major')
        self.ax1.tick_params(length=6, width=2., which='minor')
        self.ax1.xaxis.set_ticks_position('bottom')
        self.ax1.yaxis.set_ticks_position('left')
        self.ax1.set_xlim(0.,2.5)
        self.ax1.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax1.xaxis.set_major_locator(MultipleLocator(.5))
        xlabels = [item.get_text() for item in self.ax1.get_xticklabels()]
        empty_string_labels = ['']*len(xlabels)
        self.ax1.set_xticklabels(empty_string_labels)

        self.ax2.set_xlabel(r'Ratio', fontsize=fs)
        self.ax2.set_ylabel(r'$\mathrm{log\,(Count)}$', fontsize=fs)
        self.ax2.tick_params(axis='y', which='major', labelsize=fs, pad=16)      
        self.ax2.tick_params(axis='x', which='major', labelsize=fs, pad=16)
        self.ax2.tick_params(length=12, width=2., which='major')
        self.ax2.tick_params(length=6, width=2., which='minor')
        self.ax2.xaxis.set_ticks_position('bottom')
        self.ax2.yaxis.set_ticks_position('left')
        self.ax2.set_xlim(0.,2.5)
        self.ax2.xaxis.set_minor_locator(MultipleLocator(.1))
        self.ax2.xaxis.set_major_locator(MultipleLocator(.5))

        plt.subplots_adjust(
          bottom=0.085, top=0.98, left=0.15, right=.95, hspace=.05)

    def read_data(self):

        inp_dir = os.path.join(self.top_dir, 'input_data/train')
        #for fname in os.listdir(inp_dir)[0:200]:
        for fname in os.listdir(inp_dir):
            try:
                img_data = load_img(os.path.join(inp_dir, fname))                    
                dim = img_to_array(img_data).shape
                r = dim[0] / dim[1]
                self.ratio.append(r)
                if r > 1:
                    self.ratio_clean.append(r)
                else:
                    self.ratio_clean.append(1. / r)
                del img_data #Delete variable to free memory.
            except:
                print (fname)

    def plot_histogram(self):
        bins = np.arange(0., 2.5, 0.05)
        self.ax1.hist(np.array(self.ratio), bins=bins)
        self.ax2.hist(np.array(self.ratio_clean), bins=bins)

        self.ax2.axvline(x=1.15, ls='--', c='k', lw=2.)
        self.ax2.axvline(x=1.5, ls='--', c='k', lw=2.)

        self.ax1.text(
          0.95, 0.9, r'Raw data', fontsize=fs, horizontalalignment='right',
          verticalalignment='center', transform=self.ax1.transAxes)

        self.ax2.text(
          0.95, 0.9, r'After flipping', fontsize=fs, horizontalalignment='right',
          verticalalignment='center', transform=self.ax2.transAxes)
        
    def manage_output(self):
        if self.save_fig:
            out_dir = os.path.join(self.top_dir, 'output_figures')
            fpath = os.path.join(out_dir, 'Fig_ratio_hist.pdf')
            plt.savefig(fpath, format='pdf', rasterized=True)
        if self.show_fig:
            plt.show() 

    def run_plot(self):
        self.set_fig_frame()
        self.read_data()
        self.plot_histogram()
        self.manage_output()    

if __name__ == '__main__':
    Make_Fig(show_fig=False, save_fig=True)
