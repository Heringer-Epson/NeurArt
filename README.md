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
conda install -c conda-forge tensorflow
```
Once you have your conda environment set up and activated, git clone this
package.
```
git clone git@github.com:Heringer-Epson/NeurArt.git
```

### Technical details

This section will be written soon and will include:

1. Data collection.
2. Data pre-processing.
3. Description of the neural network and its parameters.

### Directory tree
.
| src
|-- format_images.py
|-- master.py
|-- preproc_data.py
|-- train_model.py
| output_data
|-- weights.h5

### Project phases

1. Project desing, proof-of-concept and data-preprocessing (completed).
2. Prototype package, with optimized parameters and user-friendly (current).
3. Move project to Azure and expand database and number of styles (upcoming).



