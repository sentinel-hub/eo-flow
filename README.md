# EOFlow

This repository provides code and examples for creation of Earth Observation (EO) projects using TensorFlow. The code uses TensorFlow 2.0 with Keras as the main model building API.

Common model architectures, layers, and input methods for EO tasks are provided in the package `eoflow`. Custom models and input methods can also be implemented building on top of the provided abstract classes. This package aims at seamlessly integrate with [`eo-learn`](https://github.com/sentinel-hub/eo-learn), and favours both creation of models for prototypying as well as production of EO applications.

Architectures and examples for land cover and crop classification using time-series derived from satellite images are provided.

## Installation

The package can be installed by running the following command.
```
$ pip install git+https://github.com/sentinel-hub/eo-flow
```

You can also install the package from source. Clone the repository and run the following command in the root directory of the project.
```
$ pip install .
```

## Getting started

The `eoflow` package can be used in two ways. For best control over the workflow and faster prototyping, the package can be used programmatically (in code). The [example notebook](examples/notebook.ipynb) should help you get started with that. It demonstrates how to prepare a dataset pipeline, train the model, evaluate the model and make predictions using the trained model.

An alternate way of using `eoflow` is by writing configuration `json` files and running them using `eoflow`'s execute script. Configuration files specify and configure the task (training, evaluation, etc.) and contain the configurations of the model and input methods. Example configurations are provided in the `configs` directory. Once a configuration file is created it can be run using the execute command.

A simple example can be run using the following command. More advanced configurations are also provided.
```
$ python -m eoflow.execute configs/example.json
```

This will create an output folder `temp/experiment` containing the tensorboard logs and model checkpoints.

To visualize the logs in TensorBoard, run
```
$ tensorboard --logdir=temp/experiment
```

## Writing custom code

To get started with writing custom models and input methods for `eoflow` take a look at the example implementations ([`examples` folder](examples/)). Custom classes use schemas to define the configuration parameters in order to work with the execute script and configuration files. Since eoflow builds on top of TF2 and Keras, model building is very similar.

## Package structure

The subpackages of `eoflow` are as follows:
* `base`: this directory contains the abstract classes to build models, inputs and tasks. Any useful abstract class should go in this folder.
* `models`: classes implementing the TF models (e.g. Fully-Convolutional-Network, GANs, seq2seq, ...). These classes inherit and implement the `BaseModel` abstract class. The module also contains custom losses, metrics and layers.
* `tasks`: classes handling the configurable actions that can be applied to each TF model, when using the execute script. These actions may include training, inference, exporting the model, validation, etc. The tasks inherit the `BaseTask` abstract class.
* `input`: building blocks and helper methods for loading the input data (EOPatch, numpy arrays, etc.) into a tensoflow Dataset and applying different transformations (data augmentation, patch extraction)
* `utils`: collection of utility functions

### Examples and scripts

Project also contains other folders:
* `configs`: folder containing example configurations for different models. Config parameters are stored in .json files. Results of an experiment should be reproducible by re-running the same config file. Config files specify the whole workflow (model, task, data input if required).
* `examples`: folder containing example implementations of custom models and input functions. Also contains a jupyter notebook example.

## Implemented architectures

Segmentation models for land cover semantic segmentation:
* **Fully-Convolutional-Network (FCN, a.k.a. U-net)**, vanilla implementation of method described in this [paper](https://arxiv.org/abs/1505.04597). This network expects 2D MSI images as inputs and predicts 2D label maps as output.
* **Temporal FCN**, where the whole time-series is considered as a 3D MSI volume and convolutions are performed along the temporal dimension as well spatial dimension. The output of the network is a 2D label map as in previous cases. More details can be found in this [paper](https://www.researchgate.net/publication/333262625_Spatio-Temporal_Deep_Learning_An_Application_to_Land_Cover_Classification).
* **ResUNet-a**, architecture proposed in Diakogiannis et al. ["ResUNet-a: A deep learning framework for semantic segmetnation of remotely sensed data"](https://www.sciencedirect.com/science/article/abs/pii/S0924271620300149). Original `mxnet` implementation can be found [here](https://github.com/feevos/resuneta).

Classification models for crop classification using time-series:
* **TCN**: Implementation of the TCN network taken from the [keras-TCN implementation by Philippe Remy](https://github.com/philipperemy/keras-tcn).
* **TempCNN**: Implementation of the TempCNN network taken from the [temporalCNN implementation of Charlotte Pelletier](https://github.com/charlotte-pel/temporalCNN).
* **Recurrent NN**: Implementation of (bidirectional) Recurrent Neural Networks for the classification of time-series. Implementation allows to use either `SimpleRNN`, `GRU` or `LSTM` layers as building blocks of the architecture.
* **TransformerEncoder**: Implementation of a time-series classification architecture based on [self-attention](https://arxiv.org/abs/1706.03762) layers. This implementation follows [this PyTorch implementation of Marc Russwurm](https://github.com/MarcCoru/crop-type-mapping). 
* **PSE+TAE**: Implementation of the Pixel-Set Encoder and temporal Self-attention proposed in Garnot V. _et al._ 
["Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention"](https://hal.archives-ouvertes.fr/hal-02879223/document). 
This implementation is adapted from the [Pytorch version](https://github.com/VSainteuf/pytorch-psetae).  

Descriptions and examples of semantic segmentation architectures are available [here](MODELS.md).
