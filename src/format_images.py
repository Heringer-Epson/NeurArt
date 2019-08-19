#!/usr/bin/env python

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import scipy

#Set relevant directories.
top_dir = os.path.abspath(
  os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
inp_dir = os.path.join(top_dir, 'input_data/train')
out_dir = os.path.join(top_dir, 'input_data/train_clean')
'''
for i, fname in enumerate(os.listdir(inp_dir)):
    
    try: #Some images may trigger a decompression error. Ignore those.
        img_data = load_img(os.path.join(inp_dir, fname))                    
        #img_data = load_img(os.path.join(inp_dir, fname),
        #                    target_size=(900, 1200))                    
        
        dim = img_to_array(img_data).shape
        r = dim[0] / dim[1] #note that dim[0] is y and dim[1] is x.

        #Only include paintings that have a standard ratio.
        if ((r > 0.6 and r < 0.9) or (r > 1.10 and r < 1.67)): 
            print (i)
        
            #Most paintings are vertical and have a ratio of 4/3. If the
            #painting is horizontal, rotate it.
            if r < 1:
                img = scipy.ndimage.rotate(img_data, 90, reshape=True)
            else:
                img = scipy.ndimage.rotate(img_data, 0, reshape=True)
        
            #Convert image array to an Image object and resize it.
            img = Image.fromarray(img)
            img = img.resize((900,1200), Image.ANTIALIAS)

            #plt.imshow(img)
            #plt.show()
            img.save(os.path.join(out_dir, fname))

            del img #Delete variables to free memory.
        del img_data
    
    except:
        pass
'''

#Read original file of attributes.
attr_filepath = os.path.join(top_dir, 'input_data/train_info.csv')
attr = pd.read_csv(attr_filepath, header=0)

#Make a clean train dictionary. This does not include files that were excluded
#in the loop above.
filenames = [fname for fname in os.listdir(out_dir)]
train_clean = pd.DataFrame({'filename':filenames})
merged_clean = pd.merge(train_clean, attr, on='filename')

#Save output cleaned dictionary.
fpath = os.path.join(top_dir, 'input_data/train_info_clean.json')
merged_clean.to_json(fpath)

