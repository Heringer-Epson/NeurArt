# NeurArt

Welcome to Neurart, a machine learning approach to classifying artworks based
on their style. Art styles are often complex and their proper classification
requires a deep understaing of not only the art features, but also the
historical context and the location where the work was cerated.

Neurart uses a large pool of pre-classified art to train a neural network model
on how to distinguish between art styles. In the current implementation the
following styles are used. Future implementations should expand on this pool.

+ Renaissance
+ Impressionism
+ Cubism
+ Ukiyo-e
+ Dada
+ Pointillism

### Usage and output:
To use Neurart, find or take a picture of your favorite art work and feed
it to the program, as shown in the demo below. After a few seconds, the
pre-trained network will decide on the three most likely art styles and
return their respective probabilities. Go to the source directory in your
cloned repository and type
```python
python3 master.py 'PATH_TO_IMAGE'
```
NeurArt will return the most likely art styles. For example
```
92.3% Impressionism
4.8% Renaissance
2.9% Cubism
```

### Requirements
The following packages are necessary for this program, which I currently
run on Ubuntu 18.04.2 LTS. See installations tips below.
```
pandas
numpy
matplotlib
sklearn
tensorflow
```

### Installation
This package was installed in the py37 conda env, enhanced with tensorflow:
```
conda create -n py37 -c anaconda python=3.7
conda install pandas
conda install -c conda-forge tensorflow
conda install matplotlib=3.1.1
conda install -c anaconda pillow=6.1.0
conda install scikit-learn=0.21.2
```
Once you have your conda environment set up and activated, git clone this
package.
```
git clone git@github.com:Heringer-Epson/NeurArt.git
```

### Input parameters
The following parameters can be changed in the src/master.py file.

*styles* : ~list (of strings)
Which styles to use in the training set. Default is ['Impressionism', 'Renaissance', 'Cubism', 'Ukiyo-e', 'Dada', 'Pointillism']. The complete list of options can be found HERE.

*img_x* : ~int
Training the neural network requires images to have standard dimensions. This variable will set the number of pixels for the image width. Default is 400.

*img_y* : ~int
Training the neural network requires images to have standard dimensions. This variable will set the number of pixels for the image height. Default is 600.

### Technical details

This section will be written soon and will include:

**Data collection**
The images used here were gathered from the [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) challenge on Kaggle. The files test.zip and train.zip are combined to yield 103,250 artworks divided in 137 styles. Corrupted files are replaced using replacements_for_corrupted_files.zip. The original data was collected from [WikiArt.org](https://www.wikiart.org/).



**Data pre-processing**
1. The usage of square canvas is uncommon for artworks. Therefore, I first inspect the distribution of canvas ratios, finding two predominant choices of height/width: 3/4 and 4/3 (see image below). In order to produce a data sample of standard dimensions, I select only the paintings with a ratio close to 4/3 and rotate those of ratio 3/4. Painting that do not conform with this choice are removed. This measure ensures that the neural network is trained and applied to paintings that have not been distorted by the choice of dimensions. The resized images are stored seprately in order to preserve disk space.

2. Rows that do not have a style assigned are removed.

These operations reduce the number of artworks to 85949.

**Description of the neural network and its parameters**

+ A data generator is created using Keras' ImageDataGenerator, where:
  + The pixel intensity is rescaled by 1./255, such that intensities are between 0 and 1.
  + A fraction of the data is used for validation (10%).
  + The data are augmented by allowing the images to be horizontally flipped.
  + The batch_size is chosen to 50. Other values produced similar results.
+ The neural network is created using the keras' Sequential function.
  + The first 2D convolution layer has 32 nodes.
  + Another 3 convolution layers are added, using 32, 64 and 64 nodes, respectively.
  + The model has then a Dropout of 0.25, followed by Flatten, a dense layer with 256 nodes, another Dropout (of 0.5 this time) and a final dense layer with as many nodes as number of styles requested.
  + The kernel size used is (3,3) and the default dimensions are height=600 pixels abd width=400 pixels. The activation is 'relu' and, to speed up the training, I adopt strides=2.
+ The model is compiled by optimizing the crossentropy and using the 'adam' optimizer and using 2 epochs.
  + The accuracy obtained is of approximetely 70%, which is quite reasonable for the complex task of classifying art styles. 

### Directory tree

```bash
NeurArt/
|-- src/
|   |-- master.py
|   |-- format_images.py
|   |-- preproc_data.py
|   |-- train_model.py
|-- analyses/
|   |-- pixel_hist.py
|   |-- style_hist.py
|-- output_data/
|   |-- weights.h5
```

### Project phases

1. Project desing, proof-of-concept and data-preprocessing (completed).
2. Prototype package, with optimized parameters and user-friendly (current).
3. Move project to Azure and expand database and number of styles (upcoming).



