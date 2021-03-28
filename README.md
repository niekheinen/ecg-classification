# ecg-classification 
## Introduction
This repository contains code written for my research project: "Using lightweight image classifiers for electorcardiogram classification on embedded devices".
More information about my research project can be found [here](http://essay.utwente.nl/82240/).
The code consists maily out of 4 parts:
#### 1. Signal reshaper
This part can be used to reshape 1D ECG signals into 3D image tensors, so that they can be used by image classifiers.

#### 2. Dataset utilities 
Functions to load the data from the MIT-BIH Arrhythmia Database so that it can be used by the signal reshaper. 
A copy of the used dataset for this project can be found in ./ecgc/mitdb/.
More information about the MIT-BIH Arryhmia Database can be found [here](https://www.physionet.org/content/mitdb/1.0.0/).

#### 3. Training notebook
The training  notebook (training.ipynb) contains the code that actually creates and trains the neural networks that classify the reshaped ECG signals.
The notebook can be run on Google Colab to decreases training times.

#### 4. Evaluation utilities
Functions to evalaute the performance of the trained networks.

## Documentation
The documentation notebook (documentation.ipynb) contains a bit of documentation and some examples about what you can do with this code.

## Installation
I used conda to manage the dependencies of this project. To install the project run:

    $ conda env create -f environment.yml

If that doesn't work try this:

    $ conda create --name ecg-classification --file requirements.txt
