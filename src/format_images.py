#!/usr/bin/env python
import sys, os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import scipy

def crop_and_reduce():
    """Routine to pre-process the original paintings. The following tasks are
    executed:
    
    1) Remove images that are obnouxiously large. Done with try/except.
    2) Generalize dimensions.
      - Rotate landscapes so that all images appear as portraits.
      - Remove paintings whose ratios are not close to 4:3
      - Resize remaining images such that width=900 and height=1200 pixels.
    3) Convert colors to grayscale.
    This is reduces the size of the images and allows for larger batches when
    training the neural network. The assumption here is that the color
    information is not necessary to distinguish between art-styles.
    
    Outputs:
    --------
    './../input_data/train_updated/X.jpg where X are the image names.'
    """

    #Set relevant directories.
    inp_dir = os.path.join('./../input_data/', 'raw_images')
    out_dir = os.path.join('./../input_data/', 'train_updated')

    #Iterate through all images in the input directory.
    for i, fname in enumerate(os.listdir(inp_dir)):
                    
        #Some images may trigger a decompression error. Ignore those.
        try:
            
            #Load image.
            img_data = load_img(os.path.join(inp_dir, fname))                                    
            
            #Record image dimensions (width and height in number of pixels).
            dim = img_to_array(img_data).shape
            r = dim[0] / dim[1] #note that dim[0] is y and dim[1] is x.

            #Only include paintings that have a standard ratio r.
            if ((r > 0.6 and r < 0.9) or (r > 1.10 and r < 1.67)): 
                print (i)
            
                #Most paintings are vertical and have a ratio of 4/3. If the
                #painting is horizontal, rotate it.
                if r < 1:
                    img = scipy.ndimage.rotate(img_data, 90, reshape=True)
                else:
                    img = scipy.ndimage.rotate(img_data, 0, reshape=True)
            
                #Convert image array to an Image object, converto to grayscale,
                #resize and save it.
                img = Image.fromarray(img)
                img = img.convert('L')
                img = img.resize((900,1200), Image.ANTIALIAS)
                img.save(os.path.join(out_dir, fname))
                del img #Delete variables to free memory.
            del img_data

        except:
            pass

if __name__ == '__main__':
    crop_and_reduce()
