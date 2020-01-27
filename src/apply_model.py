#!/usr/bin/env python

import os
import scipy
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class Apply_Model(object):
    """
    Description:
    ------------
    [Under Development].
    This code retrieves the weights of the previously trained Convolutional
    Neural Network (CNN) and applies it to new input images according to the
    following steps:
    
    1) Input images are cropped and transformed to black-and-white images.
    Images that cannot be cropped while preserving its original ratio are not
    included.
    2) Load the pre-trained CNN weights and build the neural network.
    3) Produce a prediction for each input image. The prediction should include
    the 3 most likely styles and their respective probabilities.
    
    Return:
    -------
    None
    """      
    def __init__(self, img_x, img_y, styles, top_dir):
        self.img_x, self.img_y = img_x, img_y
        self.styles = styles        
        self.top_dir = top_dir
       
        self.form_df = None
        self.model = None
        self.run_application()
        
    def load_input_data(self):
        
        out_fname, out_data = [], []
        inp_dir = os.path.join(self.top_dir, 'input_images')
        for fname in os.listdir(inp_dir):

            img_data = load_img(os.path.join(inp_dir, fname))                                  
            dim = img_to_array(img_data).shape
            r = dim[0] / dim[1] #note that dim[0] is y and dim[1] is x.

            #Only include paintings that have a standard ratio.
            if ((r > 0.6 and r < 0.9) or (r > 1.10 and r < 1.67)):                
                #Most paintings are vertical and have a ratio of 4/3. If the
                #painting is horizontal, rotate it.
                if r < 1:
                    img = scipy.ndimage.rotate(img_data, 90, reshape=True)
                else:
                    img = scipy.ndimage.rotate(img_data, 0, reshape=True)
            
                #Convert image array to an Image object and resize it.
                #Note that the arg for resize is (width,height).
                img = Image.fromarray(img)
                img = img.resize((self.img_x,self.img_y), Image.ANTIALIAS)
                out_fname.append(fname)
                out_data.append(np.array(img))

                del img #Delete variables to free memory.
            del img_data
        
        self.form_df = pd.DataFrame.from_dict({'filename': out_fname,
                                               'data': out_data})
    
    def build_pretrained_model(self):
        inp_dir = os.path.join(self.top_dir, 'output_data')
        fpath = os.path.join(inp_dir, 'model.h5')
        self.model = tf.keras.models.load_model(fpath)

    def apply_model_to_input(self):

        labels = sorted(self.styles) #Tensorflow output follows sorted target.
        X = [img_array / 255. for img_array in  self.form_df['data'].values]
        pred = self.model.predict(np.asarray(X), batch_size=1, steps=1)
        predicted_class_indices=np.argmax(pred,axis=1)
        predictions = [labels[k] for k in predicted_class_indices]
        results=pd.DataFrame({'filename': self.form_df['filename'],
                              'Predicted_style': predictions})

        #Each column in the output predictions corresponds to the probability
        #of the painting being of a given styles.
        aux_pred = np.transpose(pred)
        for i in range(len(self.styles)):
            results[labels[i]] = aux_pred[i] #list of probs for a style.      

        fpath = os.path.join(self.top_dir, 'output_data/results.csv')
        results.to_csv(fpath,index=False,header=True)

        if len(results.index) < 10:#If small input, print output as well.
            fnames = results['filename'].values
            for i, p in enumerate(pred):
                sorted_idx = np.argsort(p)[::-1] #In descending order.
                print ('\n' + fnames[i] + ': ')
                for k in range(min(len(labels),3)):
                    idx = sorted_idx[k]
                    print ('  ' + labels[idx] + ' (%4.1f%%)' % (p[idx] * 100.))

    def run_application(self):
        self.load_input_data()
        self.build_pretrained_model()
        self.apply_model_to_input()
