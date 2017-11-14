.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel Corporation
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Intel® Nervana™ graph
*********************

Intel® Nervana™ graph is Intel's library for developing frameworks that can efficiently run deep learning computations on a variety of compute platforms. It consists of three primary API components:

- An API for creating computational *Intel Nervana graphs*.
- Two higher level frontend APIs (TensorFlow* and neon™) utilizing the Intel Nervana graph API for common deep learning workflows.
- A transformer API for compiling and executing these graphs.

For more information, refer to the `blog post <https://www.intelnervana.com/intel-nervana-graph-preview-release/?_ga=2.139466358.473888884.1509049473-747831713.1505851199/>`_ announcing our
preview release!

Installation
============

Install the base packages for the CPU backend by following the instructions in the installation documentation
`here <https://ngraph.nervanasys.com/docs/latest/installation.html>`_.

After you complete the prerequisites and install the base Intel Nervana graph package as explained in the installation documentation, you will need to install some additional packages to run
Intel Nervana Graph at optimal performance on various compute platforms.

CPU/Intel® architecture transformer
---------------------------------------

To run Intel Nervana graph with optimal performance on a CPU backend, you need to install Intel Nervana graph with Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) support:

  1. Download Intel® MKL-DNN from `here <https://github.com/01org/mkl-dnn>`_.
  2. Follow the installation instructions in the `README.md <https://github.com/01org/mkl-dnn/blob/master/README.md>`_ to install MKL-DNN. 
  3. Set the environment variable `MKLDNN_ROOT` to point to the location where you installed Intel MKL-DNN.
  
  ::
  
    export MKLDNN_ROOT=/path/to/mkldnn/root

GPU transformer
---------------

To run Intel Nervana graph on a GPU backend, you need to install CUDA* and then enable the GPU transformer:

  1. Download and install CUDA according to the `instructions <http://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>`_.
  2. On your system, enable the GPU transformer::

    make gpu_prepare
    

Virtual environment activation
==================================

The virtual environment for Intel Nervana Graph is created when you install the prerequisites described in the installation documentation
`here <https://ngraph.nervanasys.com/docs/latest/installation.html>`_ and in the `README <https://github.com/NervanaSystems/ngraph/blob/master/README.md>`_.

To activate a Python virtualenv, run the following command::
  
  . .venv/bin/activate


Build Intel Nervana graph
=========================

Run the following command within your virtual environment to build Intel Nervana graph::

  make install


Additional options
==================

Use these commands to run the tests::

  make [test_cpu|test_mkldnn|test_gpu|test_integration]


Before checking in code, ensure there are no "make style" errors by running the following check::
  
  make style


If the check returns any errors, use this command to fix style errors::
  
  make fixstyle


To generate the documentation as HTML files::
  
  sudo apt-get install pandoc
  make doc


Examples
========

- *examples/walk_through/* contains several code walk throughs.
- *examples/mnist/mnist_mlp.py* uses the neon front-end to define and train a MLP model on MNIST data.
- *examples/cifar10/cifar10_conv.py* uses the neon front-end to define and train a CNN model on CIFAR10 data.
- *examples/cifar10/cifar10_mlp.py* uses the neon front-end to define and train a MLP model on CIFAR10 data.
- *examples/ptb/char_rnn.py* uses the neon front-end to define and train a character-level RNN model on Penn Treebank data.

Training deep residual networks
===============================

This example demonstrates training a deep residual network as first described in `He et. al. msra1 <http://arxiv.org/abs/1512.03385>`_. It can handle CIFAR10 and Imagenet datasets

Files
-----

- *data.py*: Implements dataloader for CIFAR10 and imagenet dataset.
- *resnet.py*: Defines model for Residual network.
- *train_resnet.py*: Processes command line arguments, like the choice of dataset and number of layers, and trains the Resnet model.

Dataset
-------

The `CIFAR10` Dataset gets downloaded automatically to *~/*. To download and use the dataset from a specific location, set ``--data_dir i1k``.
For imagenet, update ``manifest_root`` to the location of your imagenet dataset in *data.py*. Also update ``path`` to the directory where manifest ``.csv`` files are stored in *data.py*.

Usage
-----

Use the following command to run training on Intel Nervana Graph::

  python examples/resnet/train_resnet.py -b <cpu,gpu> --size <20,56> -t 64000 -z <64,128>
  
Intel Nervana Graph uses the `CIFAR10` dataset by default. If you would like to train using a different dataset, like the ``i1k`` dataset, provide the location of the dataset as ``BASE_DATA_DIR= </path/to/load/file>`` , and then add the ``--dataset: <name of data set>`` argument to the command above. 

Citation
--------

`Deep Residual Learning for Image Recognition <http://arxiv.org/abs/1512.03385>`_

