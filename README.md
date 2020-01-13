# EOFlow

This repository provides models and tools for creation of TensorFlow earth observation projects. Common model architectures, layers, and input methods for earth observation tasks are provided in the package `eoflow`. Custom models and input methods can also be implemented building on top of the provided abstract classes. The code is based on TensorFlow 2.0 with Keras as the main model building API.

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

An alternate way of using `eoflow` is by writing configuration files and running them using eoflow's execute script. Configuration files specify and configure the task (training, evaluation, etc.) and contain the configurations of the model and input methods. Example configurations are provided in the `configs` directory. Once a configuration file is created it can be executed using the execute command.

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

## Currently implemented models

Segmentation models:
* **Fully-Convolutional-Network (FCN, a.k.a. U-net)**, vanilla implementation of method described in this [paper](https://arxiv.org/abs/1505.04597). This network expects 2D MSI images as inputs and predicts 2D label maps as output.
* **Recurrent FCN**, where a time series is used as input and the temporal dependency between images is modelled by recurrent convolutions. The output of the network is a 2D label map as in previous case.
* **Temporal FCN**, where the whole time-series is considered as a 3D MSI volume and convolutions are performed along the temporal dimension as well spatial dimension. The output of the network is a 2D label map as in previous cases.

Classification models:
* **TCN**: Implementation of the TCN network taken from the [keras-TCN implementation](https://github.com/philipperemy/keras-tcn).
* **TempCNN**: Implementation of the TempCNN network taken from the [temporalCNN implementation](https://github.com/charlotte-pel/temporalCNN).

Model descriptions and examples are available [here](MODELS.md).
