#!/usr/bin/env python

import os
#Import NEURART routines. 
from preproc_data import Preproc_Data
from train_model import Train_Model
from apply_model import Apply_Model

class Master(object):
    """
    Description:
    ------------
    This is the main code to call and execute the default routines of NeurArt.
    
    Prior to running this code, one must ensure that the original data from
    the 'Painter by Numbers' Kaggle competition has been treated in the way
    implemented in the format_images.py script:
    
    1) Corrupted images have been replaced.
    2) Images have been cropped and their ratio has been (approximately)
    preserved.
    3) Images were converted to grayscale.
    
    This master routine will:
    1) Pre-process the data by creating a pandas dataframe that contains
    information of the subset of paintings of the requested styles.
    2) Train a Convolutional Neural Network (CNN) model on this subset of
    images. The CNN model conatins 5 layes in total and adopts a cross-entropy
    loss function. Several epochs are used to expose the model suffuciently,
    so that complex, non-linear, features are captured. Each layer contains
    a Dropout method to prevent over-fitting. The trained wieghts are stored.
    3) [Under development]. The trained CNN is used to make predictions on
    new paintings. 

    Parameters:
    -----------
    img_x : ~in
        Number of pixels respective to the image width. Standard cropping
        perfomed in data pre-processing is 900.
    img_y : ~in
        Number of pixels respective to the image height. Standard cropping
        perfomed in data pre-processing is 1200.    
    train_flag : ~boolean
        Whether to train the CNN model. If the model has been previously
        trained with a satisfactory accuracy, this may be set False. 
    apply_flag : ~boolean
        Whether to apply the model to test data. This will be fully developed
        once the final product is ready for deployment. Test images should be
        stored at ./../input_images/
    styles : list of strings
        List containing art style names to be used for training the CNN.

    Useful Resources:
    -----------------
    https://www.youtube.com/watch?v=j-3vuBynnOE by sentdex.

    Return:
    -------
    None
    """  
    def __init__(self, train_flag, apply_flag, img_x, img_y, styles):
               
        if train_flag:
            Preproc_Data(styles)
            Train_Model(img_x, img_y)

        if apply_flag:
            Apply_Model(img_x, img_y, styles, top_dir)
       
if __name__ == '__main__':
    use_styles = ['Ukiyo-e', 'Cubism']
    #use_styles = ['Renaissance', 'Cubism', 'Ukiyo-e']
    Master(train_flag=True, apply_flag=False, img_x=900, img_y=1200,
           styles=use_styles)


