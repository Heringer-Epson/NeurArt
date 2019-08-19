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

Usage and Output:
To use Neurart, find or take a picture of your favorite art work and feed
it to the program, as shown in the demo below. After a few seconds, the
pre-trained network will decide on the three most likely art styles and
return their respective probabilities. 


### Requirements
This package was installed in the py37 conda env, enhanced with tensorflow:
```
conda create -n py37 -c anaconda python=3.7
conda install -c conda-forge tensorflow
```


### Installation

### Technical details

Blurb about the difficulties in classiying complex images.

the database and pre-processing
Training accuracy.

### Directories
.
> src
>> master.py
>> preproc_data.py




