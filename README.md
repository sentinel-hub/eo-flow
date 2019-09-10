# Tensorflow projects

This repository provides a templated structure to generate TensorFlow
projects. Using the same structure for all TF projects should benefit
model creation, debugging and experiments reproducibility.

## Template structure

The template structure is taken from  [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template).

The project directories are as follows:
 * `base`: this directory contains the abstract classes to build a model
 and a trainer. Any useful abstract class should go in this folder.
 * `models`: classes implementing the TF models (e.g. Fully-Convolutional-Network,
 GANs, seq2seq, ...). These classes inherit and implement the `BaseModel` abstract
 class.
 * `trainers`: classes handling the training actions for each TF model.
 The same trainer class could be used to train different models. These
 classes inherit the `BaseTrainer` abstract class. An out-of-bag
 cross-validation evaluation is implemented, to allow evaluation of the
 models on held-out data at each training epoch. This allows to estimate
 model over-fitting during training. Currently the held-out data is the
 same for all training epochs and it's sampled without replacement.
 * `predictors`: classes to handle predictions once the model has been
 trained. The predictors accept a frozen `.pb` file and perform the
 prediction of the specified items of an `EOPatch`, and save result in
 an EOPatch.
 * `data_loader`: these classes handle the loading of the input data into
 batches. These classes are specific to the problem and data at hand.
 * `utils`: collection of utility function such as configuration file
 parser, directory creation, and logging functionality to TensorBoard.
 * `mains`: this folder holds the scripts that allow to run specific
 projects and experiments, by chaining `models`, `data_loaders` and
 `trainers`.
 * `configs`: folder storing the configuration files completely
 determining the behaviour of a project/experiment. Config parameters are
 stored in .json files. Results of an experiment should be reproducible
 by re-running the same config file.

For more details on the structure of the projects and on how to create
new models see [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template).

## Template and example files

Template files `trainers/template_trainer.py` and
`models/template_model.py` are available as guidelines on how the methods for
**trainers** and **models** should be implemented.

A toy example is available in `mains/fcn_main.py`. It uses the
`configs/example.json` configuration file, the `data_loader/data_generator.py`,
`models/example_model.py` and `trainers/example_trainer.py`.

To run the example, run
```
$ python mains/example_main.py --config=configs/example.json
```
This will create an output folder `experiments/example` with the `logs`
and `checkpoints`. To visualise the logs in TensorBoard, run
```
$ tensorboard --logdir=experiments/example/logs
```

## Currently implemented projects

The projects currently implemented are:
 * Fully-Convolutional-Network (FCN, a.k.a. U-net), vanilla
 implementation of method described in this [paper](https://arxiv.org/abs/1505.04597).
 This network expects 2D MSI images as inputs and predicts 2D label maps
 as output.
 * Recurrent FCN, where a time series is used as input and the temporal
 dependency between images is modelled by recurrent convolutions. The
 output of the network is a 2D label map as in previous case.
 * Temporal FCN, where the whole time-series is considered as a 3D MSI
 volume and convolutions are performed along the temporal dimension as
 well spatial dimension. The output of the network is a 2D label
 map as in previous cases.

### Fully-Convolutional-Network (FCN)

The vanilla architecture as in the following figure is implemented.
Convolutions are run along spatial dimensions of input tensor, which is
supposed to have `[M, H, W, D]` shape, where M is the mini-batch size,
and H, W and D are the height, width and number of bands (i.e. depth) of
the input image tensor. The 2d convolutions perform a `VALID` convolution,
therefore the output tensor size is smaller than the input size.

![FCN](./figures/fcn.png "FCN")

This model uses the `eodata_generator.py` script to handle `EOPatch`
objects. If the data in the _eopatch_ is a time series, a single temporal
 frame needs to be specified.

Scripts needed to train a FCN are:
 * `models/fcn_model.py` and `models/layers.py`
 * `trainers/fcn_trainer.py`
 * `data_loader/eodata_generator.py`
 * `mains/fcn_main.py`

To train a **FCN**, customise and rename the `configs/fcn_test.json` file and
run
```
python mains/fcn_main.py --config=configs/<your_fcn_config>.json
```

The `fcn_test.json` example can be run by using the `eopatches` stored in
`eo_data/dutch_arable_land_patches/traindata` on `azrael`.

### Recurrent Fully-Convolutional-Network (RFCN)

A recurrent version of the **FCN** is implemented as in below figure. The
input tensor in this case is 5d with shape `[M, T, H, W, D]`, where `T` is the
number of temporal acquisitions. As for the FCN, the 2d convolutions
operate along the `H` and `W` dimensions. The recurrent layers are applied
along the skip connections and the bottom layers to model the temporal
relationship between the features extracted by the 2d convolutions. The
output of the recurrent layers is a 4d tensor of shape `[M, H, W, D]` (the
height, width and depth of the tensors will vary along the network). The decoding path is
as in **FCN**.
The 2d convolutions perform a `VALID` convolution,
therefore the output tensor size is smaller than the input size.

![RFCN](./figures/rfcn.png "RFCN")

This model uses the `EOMultiTempDataGenerator` class in `eodata_generator.py`
to handle `EOPatch` time-series.

Scripts needed to train a RFCN are:
 * `models/rfcn_model.py` and `models/layers.py`
 * `trainers/fcn_trainer.py`
 * `data_loader/eodata_generator.py`
 * `mains/rfcn_main.py`

To train a **RFCN**, customise and rename the `configs/rfcn_test.json` file and
run
```
python mains/rfcn_main.py --config=configs/<your_rfcn_config>.json
```

The `rfcn_test.json` example can be run by using the `eopatches` stored in
`eo_data/dutch_arable_land_patches/traindata` on `azrael`.

### Temporal Fully-Convolutional-Network (TFCN)

Similarly to the RFCN, the TFCN works with time-series of input shape
`[M, T, H, W, D]`. This network performs 3d convolutions along the
tempo-spatial dimensions, i.e. the convolutional kernels are 3d `k x k x k`.
As default, the temporal dimension is not pooled. For temporal pooling, enough
time-frames need to be available in the input tensors. At the bottom of the
TFCN and along the skip connections, a 1d convolution along the temporal
dimension is performed to linearly combine the temporal features. The
resulting tensors are 4d of shape `[M, H, W, D]`. The decoding path is
as in FCN.

![TFCN](./figures/tfcn.png "TFCN")

This model uses the `EOMultiTempDataGenerator` class in `eodata_generator.py`
to handle `EOPatch` time-series.

Scripts needed to train a TFCN are:
 * `models/tfcn_model.py` and `models/layers.py`
 * `trainers/fcn_trainer.py`
 * `data_loader/eodata_generator.py`
 * `mains/tfcn_main.py`

To train a **TFCN**, customise and rename the `configs/tfcn_test.json` file and
run
```
python mains/tfcn_main.py --config=configs/<your_tfcn_config>.json
```

The `tfcn_test.json` example can be run by using the `eopatches` stored in
`eo_data/dutch_arable_land_patches/traindata` on `azrael`.

## Create your own project

In order to create your own project, you need to create a **model** and a
**trainer** if not already available. The **model** should import the
`models/layers.py` module, and if layers do not yet exist, should be added
to the module. This should facilitate re-use and clarity.

Depending on the task at hand (i.e. classification, regression, clustering)
and data source, you might have to create a specific **data_generator**
module.

## Collection of TF projects

A collection of TensorFlow projects is available in the
[Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) GitHub
repo.